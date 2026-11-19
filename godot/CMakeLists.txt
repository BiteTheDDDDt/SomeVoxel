include(FetchContent)
FetchContent_Declare(godot_cpp
    URL https://codeload.github.com/godotengine/godot-cpp/zip/58d1de720b8ffe9f8ffcdfe3a85148582cfd2e74
    URL_HASH SHA256=ebc66a60a911936d42e5492116301b81bb97d4ccd739f1c4b0236544258bd2e0
    DOWNLOAD_EXTRACT_TIMESTAMP FALSE
)

set(GODOTCPP_TARGET template_release CACHE STRING "Godot binding target")
set(GODOTCPP_BUILD_PROFILE "${CMAKE_CURRENT_SOURCE_DIR}/build_profile.json" CACHE FILEPATH "Godot binding profile")
FetchContent_MakeAvailable(godot_cpp)
if(MINGW AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # The pinned bindings use the C allocation functions in this translation unit.
    set_property(SOURCE "${godot_cpp_SOURCE_DIR}/src/godot.cpp"
        TARGET_DIRECTORY godot-cpp APPEND PROPERTY COMPILE_OPTIONS -include cstdlib)
endif()

add_library(somevoxel_godot SHARED extension.cpp)
target_link_libraries(somevoxel_godot PRIVATE SomeVoxel::Core godot::cpp)
set_target_properties(somevoxel_godot PROPERTIES OUTPUT_NAME somevoxel)
if(WIN32)
    set_target_properties(somevoxel_godot PROPERTIES PREFIX "")
endif()
if(MINGW)
    target_link_options(somevoxel_godot PRIVATE -static)
endif()

install(TARGETS somevoxel_godot
    RUNTIME DESTINATION addons/somevoxel/bin
    LIBRARY DESTINATION addons/somevoxel/bin
)
install(FILES somevoxel.gdextension DESTINATION addons/somevoxel)
install(FILES presentation/sand_instances.gdshader DESTINATION addons/somevoxel/presentation)
install(FILES gpu/connected_heads.gd gpu/connected_heads.glsl
    DESTINATION addons/somevoxel/gpu)
install(FILES "${godot_cpp_SOURCE_DIR}/LICENSE.md" DESTINATION addons/somevoxel RENAME godot-cpp-LICENSE.md)
