project('Descale', 'c',
    default_options: ['buildtype=release', 'b_ndebug=if-release', 'c_std=c99'],
    meson_version: '>=0.46.0',
    version: '3'
)

vs = dependency('vapoursynth').partial_dependency(compile_args: true, includes: true)

sources = ['src/descale.c']

libs = []

if host_machine.cpu_family().startswith('x86')
    add_project_arguments('-DDESCALE_X86', '-mfpmath=sse', '-msse2', language : 'c')

    sources += ['src/x86/cpuinfo_x86.c']

    libs += static_library('descale_avx2', 'src/x86/descale_avx2.c',
                dependencies: [vs],
                c_args: ['-mavx2', '-mfma'],
                pic: true,
                include_directories: 'src'
            )

endif


shared_module('descale', sources,
    dependencies: [vs],
    include_directories: 'src',
    link_with: libs,
    name_prefix: 'lib',
    install: true,
    install_dir: join_paths(vs.get_pkgconfig_variable('libdir'), 'vapoursynth')
)
