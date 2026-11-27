idf_component_register(
    SRCS "mfcc_dsp.c" "noise_sup.c"
    INCLUDE_DIRS "include"
    REQUIRES freertos esp-dsp
)
