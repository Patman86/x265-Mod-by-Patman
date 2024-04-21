include(FindPackageHandleStandardArgs)

# Check if Armv8.6 Neon I8MM is supported by the Arm CPU
if(APPLE)
    execute_process(COMMAND sysctl -a
                    COMMAND grep "hw.optional.arm.FEAT_I8MM: 1"
                    OUTPUT_VARIABLE has_i8mm
                    ERROR_QUIET
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
else()
    execute_process(COMMAND cat /proc/cpuinfo
                    COMMAND grep Features
                    COMMAND grep i8mm
                    OUTPUT_VARIABLE has_i8mm
                    ERROR_QUIET
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

if(has_i8mm)
    set(CPU_HAS_NEON_I8MM 1)
endif()
