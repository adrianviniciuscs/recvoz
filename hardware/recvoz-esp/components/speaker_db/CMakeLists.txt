idf_component_register(
    SRCS "speaker_db.c" "gmm_ubm.c"
    INCLUDE_DIRS "include"
    REQUIRES nvs_flash
    PRIV_REQUIRES knn
)
