include(FindPackageHandleStandardArgs)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    execute_process(COMMAND cat /proc/cpuinfo
                    COMMAND grep Features
                    COMMAND grep svebitperm
                    OUTPUT_VARIABLE sve2_bitperm_version
                    ERROR_QUIET
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

if(sve2_bitperm_version)
    set(CPU_HAS_SVE2_BITPERM 1)
endif()
