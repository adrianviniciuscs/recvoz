idf_component_register(SRCS "recvoz-esp.c" "console.c" "capture.c" "recognition.c" "command_args.c"
    INCLUDE_DIRS "."
    REQUIRES recvoz_display audio_in mfcc speaker_db console esp_driver_gpio esp_timer)
