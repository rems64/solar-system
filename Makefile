CXXFLAGS = -g
GLFLAGS = -lglfw -lGL

SHADERS_PREFIX = bin/shaders/
SHADERS_NAMES = atmosphere.frag base.vert compositing.vert compositing.frag shading.vert shading.frag flat_clouds.frag sun.frag textured_transparent.frag textured.frag
SHADERS = $(addprefix $(SHADERS_PREFIX),$(SHADERS_NAMES))

BINDIR = bin

all: bin/main

bin/main : src/main.cpp lib/glad.o | $(SHADERS) $(BINDIR)
	@echo $(SHADERS)
	$(CXX) src/main.cpp lib/glad.o $(GLFLAGS) $(CXXFLAGS) -I include/ -o $@

lib/glad.o : src/glad.c
	$(CXX) src/glad.c -I include/ -c -o lib/glad.o

bin/shaders/%.vert: shaders/%.vert | $(BINDIR)
	cp shaders/$*.vert bin/shaders/$*.vert

bin/shaders/%.frag: shaders/%.frag | $(BINDIR)
	cp shaders/$*.frag bin/shaders/$*.frag

# Create the build directory
$(BINDIR):
	mkdir -p $@
	mkdir -p $@/shaders

run: bin/main
	./bin/main

clean::
	@rm -rf bin