all: bin/main

CXX = g++
CXXFLAGS = -g
GLFLAGS = -lglfw -lGL

bin/% : src/%.cpp lib/glad.o
	$(CXX) src/$*.cpp lib/glad.o $(GLFLAGS) $(CXXFLAGS) -I include/ -o $@

lib/glad.o : src/glad.c
	g++ src/glad.c -I include/ -c -o lib/glad.o

# sphere_shader:
# 	cp shaders/sphere.vert bin/shaders/sphere.vert
# 	cp shaders/sphere.frag bin/shaders/sphere.frag

run: bin/main
	./bin/main
