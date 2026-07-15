include(FindPackageHandleStandardArgs)

if(LOONGARCH64)

    set(CHECK_LSX_CODE "
        int main(int argc, char **argv) {
            __asm__ volatile (
               \"vadd.w $vr0, $vr1, $vr1\"
            );
            return 0; }")

    check_cxx_source_compiles("${CHECK_LSX_CODE}" SUPPORTS_LSX)

endif()
