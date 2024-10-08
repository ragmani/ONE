nnas_find_package(FlatBuffers EXACT 23.5.26 QUIET)

if(NOT FlatBuffers_FOUND)
  message(STATUS "mio-circle08 skip: FlatBuffers 23.5.26 NOT FOUND")
  return()
endif(NOT FlatBuffers_FOUND)

message(STATUS "Build mio-circle08: TRUE")

# TODO Find a better way
set(SCHEMA_FILE "${NNAS_PROJECT_SOURCE_DIR}/nnpackage/schema/circle_schema.fbs")

# NOTE Copy circle_schema.fbs as schema.fbs to generate "schema_generated.fbs" instead of "circle_schema_generated.fbs"
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/schema.fbs"
  COMMAND ${CMAKE_COMMAND} -E copy "${SCHEMA_FILE}" schema.fbs
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  DEPENDS "${SCHEMA_FILE}"
)

FlatBuffersMuteable_Target(mio_circle08
  OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/gen/mio/circle"
  INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/gen"
  SCHEMA_DIR "${CMAKE_CURRENT_BINARY_DIR}"
  SCHEMA_FILES "schema.fbs"
)

# This example shows how to use "mio-circle08" library
add_executable(mio_circle08_example example.cpp)
target_link_libraries(mio_circle08_example mio_circle08)

file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE TESTS "src/*.test.cpp")
list(REMOVE_ITEM SOURCES ${TESTS})

add_library(mio_circle08_helper STATIC ${SOURCES})
set_target_properties(mio_circle08_helper PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(mio_circle08_helper PRIVATE src)
target_include_directories(mio_circle08_helper PUBLIC include)
target_link_libraries(mio_circle08_helper mio_circle08)

if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

nnas_find_package(GTest REQUIRED)

GTest_AddTest(mio_circle08_helper_test ${TESTS})
target_include_directories(mio_circle08_helper_test PRIVATE src)
target_link_libraries(mio_circle08_helper_test mio_circle08)
target_link_libraries(mio_circle08_helper_test mio_circle08_helper)
