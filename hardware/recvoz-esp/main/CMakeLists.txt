idf_component_register(SRCS "recvoz-esp.c"
                    INCLUDE_DIRS "."
                    REQUIRES recvoz_display audio_in mfcc speaker_db console esp_driver_gpio)
