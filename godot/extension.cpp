#include "voxel_world.hpp"
#include <godot_cpp/godot.hpp>

extern "C" GDExtensionBool GDE_EXPORT somevoxel_init(
    GDExtensionInterfaceGetProcAddress get_proc_address,
    GDExtensionClassLibraryPtr library, GDExtensionInitialization *initialization) {
    godot::GDExtensionBinding::InitObject init(get_proc_address, library, initialization);
    init.register_initializer([](godot::ModuleInitializationLevel level) {
        if (level == godot::MODULE_INITIALIZATION_LEVEL_SCENE)
            godot::ClassDB::register_class<somevoxel::VoxelWorld>();
    });
    init.register_terminator([](godot::ModuleInitializationLevel) {});
    init.set_minimum_library_initialization_level(godot::MODULE_INITIALIZATION_LEVEL_SCENE);
    return init.init();
}
