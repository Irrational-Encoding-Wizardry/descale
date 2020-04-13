/*
 * This is taken from zimg
 * https://github.com/sekrit-twc/zimg/blob/b6334acf4112877afc400cd9a0ace54e171e3f11/src/zimg/common/x86/cpuinfo_x86.h
 */


#ifdef DESCALE_X86

#ifndef DESCALE_CPUINFO_X86_H
#define DESCALE_CPUINFO_X86_H


/*
 * Bitfield of selected x86 feature flags.
 */
struct X86Capabilities {
    unsigned sse             : 1;
    unsigned sse2            : 1;
    unsigned sse3            : 1;
    unsigned ssse3           : 1;
    unsigned fma             : 1;
    unsigned sse41           : 1;
    unsigned sse42           : 1;
    unsigned avx             : 1;
    unsigned f16c            : 1;
    unsigned avx2            : 1;
    unsigned avx512f         : 1;
    unsigned avx512dq        : 1;
    unsigned avx512ifma      : 1;
    unsigned avx512cd        : 1;
    unsigned avx512bw        : 1;
    unsigned avx512vl        : 1;
    unsigned avx512vbmi      : 1;
    unsigned avx512vbmi2     : 1;
    unsigned avx512vnni      : 1;
    unsigned avx512bitalg    : 1;
    unsigned avx512vpopcntdq : 1;
    /* AMD architectures needing workarounds. */
    unsigned xop : 1;
    unsigned piledriver : 1;
    unsigned zen1 : 1;
    unsigned zen2 : 1;
};


/*
 * Get the x86 feature flags on the current CPU.
 *
 * @return capabilities
 */
struct X86Capabilities query_x86_capabilities(void);


#endif  // DESCALE_CPUINFO_X86_H

#endif  // DESCALE_X86
