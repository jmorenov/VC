CXX = g++
OBJ = obj
INC = include
BIN = bin
LIB = lib
SRC = src
CXXFLAGS =-Wall -g -c -O1
OPENCV = `pkg-config opencv --cflags --libs`
LIBS = $(OPENCV)

trabajo1:$(OBJ)/trabajo1.o 
	$(CXX) -o trabajo1 $^ $(LIBS)
$(OBJ)/trabajo1.o: $(SRC)/trabajo1.cpp
	$(CXX) $(CXXFLAGS) $(SRC)/trabajo1.cpp -o $(OBJ)/trabajo1.o -I$(INC)


clean:
	 rm *~; rm $(SRC)/*~; rm $(INC)/*~; rm $(OBJ)/*;
