CXX = g++
OBJ = obj
INC = include
BIN = bin
LIB = lib
SRC = src
CXXFLAGS =-Wall -g -c -O1
OPENCV = `pkg-config opencv --cflags --libs`
LIBS = $(OPENCV)

main:$(OBJ)/main.o $(OBJ)/trabajo1.o 
	$(CXX) -o main $^ $(LIBS)
$(OBJ)/main.o: $(SRC)/main.cpp $(INC)/trabajo1.h
	$(CXX) $(CXXFLAGS) $(SRC)/main.cpp -o $(OBJ)/main.o -I$(INC)
$(OBJ)/trabajo1.o: $(SRC)/trabajo1.cpp $(INC)/trabajo1.h
	$(CXX) $(CXXFLAGS) $(SRC)/trabajo1.cpp -o $(OBJ)/trabajo1.o -I$(INC)


clean:
	 rm *~; rm $(SRC)/*~; rm $(INC)/*~; rm $(OBJ)/*; find . -executable -exec rm '{}' \;

Gitclean:
	 git rm *~; git rm $(SRC)/*~; git rm $(INC)/*~; git rm $(OBJ)/*;
