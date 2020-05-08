/*
 * This is taken from zimg
 * https://github.com/sekrit-twc/zimg/blob/b6334acf4112877afc400cd9a0ace54e171e3f11/src/zimg/common/x86/cpuinfo_x86.cpp
 */


#ifdef DESCALE_X86


#if defined(_MSC_VER)
    #include <intrin.h>
#elif defined(__GNUC__)
    #include <cpuid.h>
#endif

#include "x86/cpuinfo_x86.h"


/**
 * Execute the CPUID instruction.
 *
 * @param regs array to receive eax, ebx, ecx, edx
 * @param eax argument to instruction
 * @param ecx argument to instruction
 */
static void do_cpuid(int regs[4], int eax, int ecx)
{
#if defined(_MSC_VER)
    __cpuidex(regs, eax, ecx);
#elif defined(__GNUC__)
    __cpuid_count(eax, ecx, regs[0], regs[1], regs[2], regs[3]);
#else
    regs[0] = 0;
    regs[1] = 0;
    regs[2] = 0;
    regs[3] = 0;
#endif
}


/**
 * Execute the XGETBV instruction.
 *
 * @param ecx argument to instruction
 * @return (edx << 32) | eax
 */
static unsigned long long do_xgetbv(unsigned ecx)
{
#if defined(_MSC_VER)
    return _xgetbv(ecx);
#elif defined(__GNUC__)
    unsigned eax, edx;
    __asm("xgetbv" : "=a"(eax), "=d"(edx) : "c"(ecx) : );
    return ((unsigned long long)(edx) << 32) | eax;
#else
    return 0;
#endif
}


struct X86Capabilities query_x86_capabilities(void)
{
    struct X86Capabilities caps = { 0 };
    unsigned long long xcr0 = 0;
    int regs[4] = { 0 };

    do_cpuid(regs, 1, 0);
    caps.sse      = !!(regs[3] & (1U << 25));
    caps.sse2     = !!(regs[3] & (1U << 26));
    caps.sse3     = !!(regs[2] & (1U << 0));
    caps.ssse3    = !!(regs[2] & (1U << 9));
    caps.fma      = !!(regs[2] & (1U << 12));
    caps.sse41    = !!(regs[2] & (1U << 19));
    caps.sse42    = !!(regs[2] & (1U << 20));

    // osxsave
    if (regs[2] & (1U << 27))
    xcr0 = do_xgetbv(0);

    // XMM and YMM state.
    if ((xcr0 & 0x06) == 0x06) {
    caps.avx  = !!(regs[2] & (1U << 28));
    caps.f16c = !!(regs[2] & (1U << 29));
    }

    do_cpuid(regs, 7, 0);
    if ((xcr0 & 0x06) == 0x06) {
    caps.avx2 = !!(regs[1] & (1U << 5));
    }

    // ZMM state.
    if ((xcr0 & 0xE0) == 0xE0) {
    caps.avx512f         = !!(regs[1] & (1U << 16));
    caps.avx512dq        = !!(regs[1] & (1U << 17));
    caps.avx512ifma      = !!(regs[1] & (1U << 21));
    caps.avx512cd        = !!(regs[1] & (1U << 28));
    caps.avx512bw        = !!(regs[1] & (1U << 30));
    caps.avx512vl        = !!(regs[1] & (1U << 31));
    caps.avx512vbmi      = !!(regs[2] & (1U << 1));
    caps.avx512vbmi2     = !!(regs[2] & (1U << 6));
    caps.avx512vnni      = !!(regs[2] & (1U << 11));
    caps.avx512bitalg    = !!(regs[2] & (1U << 12));
    caps.avx512vpopcntdq = !!(regs[2] & (1U << 14));
    }

    // Extended processor info.
    do_cpuid(regs, 0x80000001U, 0);
    caps.xop = !!(regs[2] & (1U << 11));

    // Zen1 vs Zen2.
    do_cpuid(regs, 0, 1);
    if (regs[1] == 0x68747541U && regs[3] == 0x69746E65U && regs[2] == 0x444D4163U /* AuthenticAMD */) {
    unsigned model;
    unsigned family;

    do_cpuid(regs, 1, 0);
    model  = (regs[0] >> 4) & 0x0FU;
    family = (regs[0] >> 8) & 0x0FU;

    if (family == 6) {
        family += ((regs[0] >> 20) & 0x0FU);
    } else if (family == 15) {
        family += ((regs[0] >> 20) & 0x0FU);
        model  += ((regs[0] >> 16) & 0x0FU) << 4;
    }

    caps.piledriver = family == 0x15 && model == 0x02;
    caps.zen1 = family == 0x17 && model <= 0x1FU;
    caps.zen2 = family == 0x17 && model >= 0x20U;
    }

    return caps;
}


#endif  // DESCALE_X86
