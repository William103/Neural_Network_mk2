TARGET=driver
CXX=g++
OFILES=network.o
CXXFLAGS=-g

$(TARGET): $(OFILES) $(TARGET).o
	ctags ./*
	$(CXX) $(CXXFLAGS) -o $@ $(TARGET).o $(OFILES)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

PHONY: clean run all

clean:
	rm *.o $(TARGET) tags

run:
	./$(TARGET)

all:
	make $(TARGET)
