CXXFLAGS = -g
GLFLAGS = -lglfw -lGL

# LIBAVCODECS = 
LIBAVCODECS = -lavcodec -lavformat -lavutil -lswscale

SHADERS_PREFIX = bin/shaders/
SHADERS_NAMES = atmosphere.frag base.vert compositing.vert compositing.frag shading.vert shading.frag sun.frag textured.frag
SHADERS = $(addprefix $(SHADERS_PREFIX),$(SHADERS_NAMES))

BINDIR = bin

IMGUI_RAD = imgui.cpp imgui_draw.cpp imgui_tables.cpp imgui_widgets.cpp backends/imgui_impl_opengl3.cpp backends/imgui_impl_glfw.cpp
IMGUI_SRC = $(addprefix lib/imgui/,$(IMGUI_RAD))
IMGUI_TMP := $(subst backends/,,$(IMGUI_RAD))
IMGUI_OBJ := $(addprefix lib/,$(subst .cpp,.o,$(IMGUI_TMP)))

all: bin/main

bin/main : src/main.cpp lib/glad.o $(IMGUI_OBJ) | $(SHADERS) $(BINDIR)
	@echo $(SHADERS)
	$(CXX) src/main.cpp lib/glad.o $(IMGUI_OBJ) $(GLFLAGS) $(LIBAVCODECS) $(CXXFLAGS) -I include/ -I lib/imgui/ -I lib/imgui/backends/ -o $@

lib/glad.o : src/glad.c
	$(CXX) src/glad.c -I include/ -c -o lib/glad.o

bin/shaders/%.vert: shaders/%.vert | $(BINDIR)
	cp shaders/$*.vert bin/shaders/$*.vert

bin/shaders/%.frag: shaders/%.frag | $(BINDIR)
	cp shaders/$*.frag bin/shaders/$*.frag

lib/%.o: lib/imgui/%.cpp
	$(CXX) $^ -c -I lib/imgui/ -o lib/$*.a

lib/%.o: lib/imgui/backends/%.cpp
	$(CXX) $^ -c -I lib/imgui/ -o lib/$*.o

# Create the build directory
$(BINDIR):
	mkdir -p $@
	mkdir -p $@/shaders

run: bin/main
	./bin/main

clean::
	@rm -rf bin
