#
# Copyright 2024 Sony Semiconductor Solutions Corp. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

force_arena_found = get_option('force_arena_found')

fs = import('fs')
os = host_machine.system()
arch = host_machine.cpu_family()

# Check the installation of the Arena SDK
if os == 'linux'
  lib_path = run_command('sh', '-c', 'ldconfig -p | grep -F "libarenac.so " | awk -F"=> " \'{print $2}\' | xargs', check: false).stdout().strip()
  arena_sdk_found = lib_path != ''
  if arena_sdk_found
    arena_sdk_root_abs = run_command('sh', '-c', 'dirname "$(dirname ' + lib_path + ')"').stdout().strip()
    arena_sdk_root_rel = fs.relative_to(arena_sdk_root_abs, meson.current_source_dir())
  endif
elif os == 'windows'
    res = run_command('reg', 'query', 'HKLM\\SOFTWARE\\Lucid Vision Labs\\Arena SDK', '/v', 'InstallFolder', check: false)
    if res.returncode() == 0
        path_line = res.stdout().strip().split('\n')[-1]
        arena_sdk_root_abs = path_line.split('REG_SZ')[1].strip()
        arena_sdk_root_rel = fs.relative_to(arena_sdk_root_abs, meson.current_source_dir())
        arena_sdk_found = true
    else
        arena_sdk_found = false
    endif
else
    arena_sdk_found = false
endif

if force_arena_found and not arena_sdk_found
  error('Arena SDK not found but force_arena_found option is enabled. Failing configuration.')
endif

if arena_sdk_found
    message('Arena SDK found: ' + arena_sdk_found.to_string())
    message('Arena SDK root: ' + arena_sdk_root_abs)
    message('Arena SDK root rel: ' + arena_sdk_root_rel)

    threads_dep = dependency('threads', required: true)
    openssl_dep = dependency('openssl', required: true, method: 'auto')
    if not openssl_dep.found()
        error('OpenSSL not found')
    endif

    if os == 'linux'
        if arch == 'x86_64'
            genicam_lib_path = join_paths(arena_sdk_root_abs, 'GenICam', 'library', 'lib', 'Linux64_x64')
            arena_lib_path = join_paths(arena_sdk_root_abs, 'lib64')
            ffmpeg_lib_path = join_paths(arena_sdk_root_abs, 'ffmpeg')
        elif arch == 'aarch64'
            genicam_lib_path = join_paths(arena_sdk_root_abs, 'GenICam', 'library', 'lib', 'Linux64_ARM')
            arena_lib_path = join_paths(arena_sdk_root_abs, 'lib')
            ffmpeg_lib_path = join_paths(arena_sdk_root_abs, 'ffmpeg')
        else
            error('Unsupported architecture: ' + arch)
        endif
    
        ld_flags = [
            '-L' + genicam_lib_path,
            '-L' + ffmpeg_lib_path,
            '-L' + arena_lib_path,
        ]

        genicam_libs = [
            '-lGCBase_gcc54_v3_3_LUCID',
            '-lGenApi_gcc54_v3_3_LUCID',
            '-lLog_gcc54_v3_3_LUCID',
            '-llog4cpp_gcc54_v3_3_LUCID',
            '-lMathParser_gcc54_v3_3_LUCID',
            '-lNodeMapData_gcc54_v3_3_LUCID',
            '-lXmlParser_gcc54_v3_3_LUCID'
        ]

        ffmpeg_libs = [
            '-lavcodec',
            '-lavformat',
            '-lavutil',
            '-lswresample'
        ]

        arena_libs = [
            '-larena',
            '-lsave',
            '-lgentl',
            '-llucidlog'
        ]

        link_args = ld_flags + genicam_libs + ffmpeg_libs + arena_libs
        compile_args = ['-Wall', '-g', '-O2', '-Wno-unknown-pragmas', '-fPIC']
    
    elif os == 'windows'
        if arch == 'x86_64'
            genicam_lib_path = join_paths(arena_sdk_root_abs, 'GenICam', 'library', 'CPP', 'lib', 'Win64_x64')
            arena_lib_path = join_paths(arena_sdk_root_abs, 'lib64', 'Arena')
        elif arch == 'x86'
            genicam_lib_path = join_paths(arena_sdk_root_abs, 'GenICam', 'library', 'CPP', 'lib', 'Win32_i86')
            arena_lib_path = join_paths(arena_sdk_root_abs, 'lib', 'Arena')
        else
            error('Unsupported architecture: ' + arch)
        endif

        compiler = meson.get_compiler('cpp')
        ld_prefix = compiler.get_id() == 'msvc' ? '/LIBPATH:' : '-L'
        ld_flags = [
            ld_prefix + genicam_lib_path,
            ld_prefix + arena_lib_path,
        ]

        genicam_libs = [
            'GCBase_MD_VC140_v3_3_LUCID.lib',
            'GenApi_MD_VC140_v3_3_LUCID.lib',
            'Log_MD_VC140_v3_3_LUCID.lib',
            'MathParser_MD_VC140_v3_3_LUCID.lib',
            'NodeMapData_MD_VC140_v3_3_LUCID.lib',
            'ResUsageStat_MD_VC140_v3_3_LUCID.lib',
            'XmlParser_MD_VC140_v3_3_LUCID.lib'
        ]

        arena_libs = [
            'Arena_v140.lib',
            'ArenaUI_v140.lib',
            'GenTL_LUCID_v140.lib',
            'lucidlog_v140.lib',
            'Save_v140.lib'
        ]

        link_args = ld_flags + genicam_libs + arena_libs
        if compiler.get_id() == 'msvc'
            compile_args = ['/W3', '/O2', '/MD', '/D_CRT_SECURE_NO_WARNINGS', '/DNOMINMAX']
        else
            compile_args = ['-Wall', '-g', '-O2', '-Wno-unknown-pragmas', '-fPIC']
        endif

    else
        error('Unsupported OS: ' + os)
    endif


    includes = [
        include_directories(join_paths(arena_sdk_root_rel, 'include', 'Arena')),
        include_directories(join_paths(arena_sdk_root_rel, 'include', 'Save')),
        include_directories(join_paths(arena_sdk_root_rel, 'include', 'GenTL')),
        include_directories(join_paths(arena_sdk_root_rel, 'GenICam', 'library', 'CPP', 'include')),
    ]

    arena_dep = declare_dependency(
        include_directories: includes,
        link_args: link_args,
        compile_args: compile_args + ['-DARENA_SDK_FOUND'],
        dependencies: [threads_dep, openssl_dep]
    )
else
    # When Arena SDK is not found, create an empty dependency
    arena_dep = declare_dependency()
    message('Arena SDK not found, continuing without it.')
endif

py = import('python').find_installation(pure: false)

triton_sources = ['cpp/binding.cpp']
triton_includes = []
if arena_sdk_found
    triton_sources += ['cpp/triton.cpp', 'cpp/IMX501Utils.cpp', 'cpp/parser.cpp']
    triton_includes += ['cpp/include', 'cpp/vendors/flatbuffers-1.11.0/include']
endif

py.extension_module(
    'triton_cpp',
    sources: triton_sources,
    include_directories: triton_includes,
    dependencies: [arena_dep],
    install: true,
    install_dir: py.get_install_dir() / 'modlib' / 'devices' / 'triton',
)