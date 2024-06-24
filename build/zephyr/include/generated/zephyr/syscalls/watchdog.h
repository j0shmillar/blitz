/* auto-generated by gen_syscalls.py, don't edit */

#ifndef Z_INCLUDE_SYSCALLS_WATCHDOG_H
#define Z_INCLUDE_SYSCALLS_WATCHDOG_H


#include <zephyr/tracing/tracing_syscall.h>

#ifndef _ASMLANGUAGE

#include <stdarg.h>

#include <zephyr/syscall_list.h>
#include <zephyr/syscall.h>

#include <zephyr/linker/sections.h>


#ifdef __cplusplus
extern "C" {
#endif

extern int z_impl_wdt_setup(const struct device * dev, uint8_t options);

__pinned_func
static inline int wdt_setup(const struct device * dev, uint8_t options)
{
#ifdef CONFIG_USERSPACE
	if (z_syscall_trap()) {
		union { uintptr_t x; const struct device * val; } parm0 = { .val = dev };
		union { uintptr_t x; uint8_t val; } parm1 = { .val = options };
		return (int) arch_syscall_invoke2(parm0.x, parm1.x, K_SYSCALL_WDT_SETUP);
	}
#endif
	compiler_barrier();
	return z_impl_wdt_setup(dev, options);
}

#if defined(CONFIG_TRACING_SYSCALL)
#ifndef DISABLE_SYSCALL_TRACING

#define wdt_setup(dev, options) ({ 	int syscall__retval; 	sys_port_trace_syscall_enter(K_SYSCALL_WDT_SETUP, wdt_setup, dev, options); 	syscall__retval = wdt_setup(dev, options); 	sys_port_trace_syscall_exit(K_SYSCALL_WDT_SETUP, wdt_setup, dev, options, syscall__retval); 	syscall__retval; })
#endif
#endif


extern int z_impl_wdt_disable(const struct device * dev);

__pinned_func
static inline int wdt_disable(const struct device * dev)
{
#ifdef CONFIG_USERSPACE
	if (z_syscall_trap()) {
		union { uintptr_t x; const struct device * val; } parm0 = { .val = dev };
		return (int) arch_syscall_invoke1(parm0.x, K_SYSCALL_WDT_DISABLE);
	}
#endif
	compiler_barrier();
	return z_impl_wdt_disable(dev);
}

#if defined(CONFIG_TRACING_SYSCALL)
#ifndef DISABLE_SYSCALL_TRACING

#define wdt_disable(dev) ({ 	int syscall__retval; 	sys_port_trace_syscall_enter(K_SYSCALL_WDT_DISABLE, wdt_disable, dev); 	syscall__retval = wdt_disable(dev); 	sys_port_trace_syscall_exit(K_SYSCALL_WDT_DISABLE, wdt_disable, dev, syscall__retval); 	syscall__retval; })
#endif
#endif


extern int z_impl_wdt_feed(const struct device * dev, int channel_id);

__pinned_func
static inline int wdt_feed(const struct device * dev, int channel_id)
{
#ifdef CONFIG_USERSPACE
	if (z_syscall_trap()) {
		union { uintptr_t x; const struct device * val; } parm0 = { .val = dev };
		union { uintptr_t x; int val; } parm1 = { .val = channel_id };
		return (int) arch_syscall_invoke2(parm0.x, parm1.x, K_SYSCALL_WDT_FEED);
	}
#endif
	compiler_barrier();
	return z_impl_wdt_feed(dev, channel_id);
}

#if defined(CONFIG_TRACING_SYSCALL)
#ifndef DISABLE_SYSCALL_TRACING

#define wdt_feed(dev, channel_id) ({ 	int syscall__retval; 	sys_port_trace_syscall_enter(K_SYSCALL_WDT_FEED, wdt_feed, dev, channel_id); 	syscall__retval = wdt_feed(dev, channel_id); 	sys_port_trace_syscall_exit(K_SYSCALL_WDT_FEED, wdt_feed, dev, channel_id, syscall__retval); 	syscall__retval; })
#endif
#endif


#ifdef __cplusplus
}
#endif

#endif
#endif /* include guard */
