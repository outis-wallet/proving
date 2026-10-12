//! What the device says about its memory, and this process's share of it, for bounding the
//! video memory a prover keeps (the GPU may also drive the user's display: when video memory ran
//! out, the display froze).
//!
//! - [`DeviceMemory::query`]: the size of the device's largest device-local heap and, where the
//!   driver reports them (`VK_EXT_memory_budget`), this process's budget and usage of it (Vulkan
//!   only, through wgpu's hal; `None` elsewhere).
//! - [`process_reserved`]: the memory this process's device holds (its allocator's blocks), from
//!   wgpu's allocator report.
//! - [`VramPolicy`]: the default bound: the whole process at most a share of the device-local heap
//!   (40%), and, with a driver budget, never closer than a headroom to it.

use crate::GpuContext;

/// A device's memory (see the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceMemory {
    /// The largest device-local heap's size, in bytes.
    pub local_heap: u64,
    /// `VK_EXT_memory_budget`: this process's budget of that heap and its usage of it, in bytes.
    pub budget: Option<u64>,
    pub usage: Option<u64>,
}

impl DeviceMemory {
    /// The device's memory, if its backend tells (Vulkan).
    #[cfg(not(any(target_arch = "wasm32", target_vendor = "apple")))]
    pub fn query(gpu: &GpuContext) -> Option<Self> {
        use ash::vk;
        // Safety: only physical-device queries on the device's own handles, which outlive the
        // guard; nothing is created or destroyed.
        unsafe {
            let hal = gpu.device.as_hal::<wgpu::hal::api::Vulkan>()?;
            let instance = hal.shared_instance().raw_instance();
            let physical = hal.raw_physical_device();
            let has_budget = instance
                .enumerate_device_extension_properties(physical)
                .ok()?
                .iter()
                .any(|e| e.extension_name_as_c_str() == Ok(ash::ext::memory_budget::NAME));
            let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
            let mut properties = vk::PhysicalDeviceMemoryProperties2::default();
            if has_budget {
                properties = properties.push_next(&mut budget);
            }
            instance.get_physical_device_memory_properties2(physical, &mut properties);
            let memory = properties.memory_properties;
            let heaps = &memory.memory_heaps[..memory.memory_heap_count as usize];
            let (index, heap) = heaps
                .iter()
                .enumerate()
                .filter(|(_, heap)| heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
                .max_by_key(|(_, heap)| heap.size)?;
            Some(Self {
                local_heap: heap.size,
                budget: has_budget.then(|| budget.heap_budget[index]),
                usage: has_budget.then(|| budget.heap_usage[index]),
            })
        }
    }

    #[cfg(any(target_arch = "wasm32", target_vendor = "apple"))]
    pub fn query(_gpu: &GpuContext) -> Option<Self> {
        None
    }
}

/// The memory this process's device holds now (its allocator's blocks), if the backend reports
/// it.
pub fn process_reserved(gpu: &GpuContext) -> Option<u64> {
    gpu.device.generate_allocator_report().map(|report| report.total_reserved_bytes)
}

/// How much video memory a process may keep (see the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VramPolicy {
    /// The whole process at most this share of the device-local heap, in percent.
    pub process_percent: u64,
    /// Without a device-local heap's size: the whole process at most this, in bytes.
    pub fallback_bytes: u64,
    /// With a driver budget: always leave this much of it, in bytes.
    pub headroom: u64,
    /// An explicit bound on what is kept besides (e.g. the resident columns), in bytes.
    pub cap: Option<u64>,
    /// What the driver holds for the process besides its allocator's blocks (its context,
    /// shader code, internal allocations), counted with them against the process's share: the
    /// RTX 3080's driver showed 270 to 390 MiB more per process than the allocator's report.
    pub context_allowance: u64,
}

impl Default for VramPolicy {
    fn default() -> Self {
        Self {
            process_percent: 40,
            fallback_bytes: 3 << 30,
            headroom: 1536 << 20,
            cap: None,
            context_allowance: 256 << 20,
        }
    }
}

/// Why [`VramPolicy::allows`] said no.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VramRefusal {
    /// Over the explicit cap.
    Cap,
    /// The process would hold more than its share of the device.
    ProcessShare,
    /// The driver's budget would have less than the headroom left.
    DriverBudget,
}

impl VramPolicy {
    /// `OUTIS_GPU_PROCESS_VRAM_PERCENT` and `OUTIS_GPU_COMPOSITION_VRAM_MIB` over the defaults.
    pub fn from_env() -> Self {
        let mut policy = Self::default();
        if let Some(percent) =
            std::env::var("OUTIS_GPU_PROCESS_VRAM_PERCENT").ok().and_then(|p| p.parse::<u64>().ok())
        {
            policy.process_percent = percent.min(100);
        }
        policy.cap = std::env::var("OUTIS_GPU_COMPOSITION_VRAM_MIB")
            .ok()
            .and_then(|mib| mib.parse::<u64>().ok())
            .map(|mib| mib << 20);
        policy
    }

    /// The most the whole process may hold on a device of this memory.
    pub fn process_bound(&self, memory: Option<&DeviceMemory>) -> u64 {
        match memory {
            Some(memory) => memory.local_heap / 100 * self.process_percent,
            None => self.fallback_bytes,
        }
    }

    /// Whether `bytes` more may be kept, `kept` being kept already under the cap, the process
    /// holding `reserved` (its allocator's report, if any) on a device of `memory`.
    pub fn allows(
        &self,
        bytes: u64,
        kept: u64,
        reserved: Option<u64>,
        memory: Option<&DeviceMemory>,
    ) -> Result<(), VramRefusal> {
        if self.cap.is_some_and(|cap| kept + bytes > cap) {
            return Err(VramRefusal::Cap);
        }
        // Without an allocator report, count what is kept as the process's.
        let reserved = reserved.unwrap_or(kept) + self.context_allowance;
        if reserved + bytes > self.process_bound(memory) {
            return Err(VramRefusal::ProcessShare);
        }
        if let Some(DeviceMemory { budget: Some(budget), usage: Some(usage), .. }) = memory
            && usage + bytes + self.headroom > *budget
        {
            return Err(VramRefusal::DriverBudget);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_bounds() {
        let memory = DeviceMemory { local_heap: 10 << 30, budget: Some(9 << 30), usage: Some(0) };
        let policy = VramPolicy::default();
        assert_eq!(policy.process_bound(Some(&memory)), (10u64 << 30) / 100 * 40);
        assert_eq!(policy.allows(2 << 30, 0, Some(1 << 30), Some(&memory)), Ok(()));
        // The context allowance counts: 4095 MiB of share, 1 GiB + 256 MiB held.
        assert_eq!(
            policy.allows((4095 - 1024 - 256 + 1) << 20, 0, Some(1 << 30), Some(&memory)),
            Err(VramRefusal::ProcessShare)
        );
        assert_eq!(
            policy.allows(3 << 30, 0, Some(1 << 30), Some(&memory)),
            Err(VramRefusal::ProcessShare)
        );
        let tight = DeviceMemory { usage: Some(7 << 30), ..memory };
        assert_eq!(
            policy.allows(1 << 30, 0, Some(1 << 30), Some(&tight)),
            Err(VramRefusal::DriverBudget)
        );
        let capped = VramPolicy { cap: Some(1 << 30), ..policy };
        assert_eq!(capped.allows(1 << 29, 1 << 29, None, Some(&memory)), Ok(()));
        assert_eq!(capped.allows(1 << 29, 1 << 30, None, Some(&memory)), Err(VramRefusal::Cap));
        assert_eq!(policy.process_bound(None), 3 << 30);
    }
}
