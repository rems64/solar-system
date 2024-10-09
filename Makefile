CXXFLAGS = -g
GLFLAGS = -lglfw -lGL

SHADERS_PREFIX = bin/shaders/
SHADERS_NAMES = atmosphere.frag base.vert deferred.vert deferred.frag flat_clouds.frag sun.frag textured_transparent.frag textured.frag
SHADERS = $(addprefix $(SHADERS_PREFIX),$(SHADERS_NAMES))

all: bin/main

.SECONDEXPANSION:
bin/main : src/main.cpp lib/glad.o $$(SHADERS)
	@echo $(SHADERS)
	$(CXX) src/main.cpp lib/glad.o $(GLFLAGS) $(CXXFLAGS) -I include/ -o $@

lib/glad.o : src/glad.c
	$(CXX) src/glad.c -I include/ -c -o lib/glad.o

bin/shaders/%.vert: shaders/%.vert
	cp shaders/$*.vert bin/shaders/$*.vert

bin/shaders/%.frag: shaders/%.frag
	cp shaders/$*.frag bin/shaders/$*.frag

run: bin/main
	./bin/main

clean::
	@rm -rf bin