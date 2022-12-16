TARGETS = train1 train2 train3 train4 train5 train6 train7 

x=128
y=128

CXXFLAGS = -fopenmp -Wall

ifeq ($(shell arch),x86_64)

CXXFLAGS1 =
CXXFLAGS2 = -O0
CXXFLAGS3 = -Os 
CXXFLAGS4 = -O2 
CXXFLAGS5 = -O3
CXXFLAGS6 = -O2 -ftree-vectorize # -fopt-info-vec-all
CXXFLAGS7 = -O2 -ftree-vectorize -ffast-math # -fopt-info-vec-all



endif

.PHONY: all
all: $(TARGETS)
	g++ test.cpp -o test $(CXXFLAGS7) -lm
# Macro to compile a flavor of ann
define cc_train
$(1): train.cpp
	g++ $(CXXFLAGS) $$< -o $$@ $(CXXFLAGS$(1:train%=%)) -DCFLAGS='"$(CXXFLAGS$(1:train%=%))"' -lm
endef

$(foreach target,$(TARGETS),$(eval $(call cc_train, $(target))))


.PHONY: clean
clean:
	rm -f $(TARGETS)test

.PHONY: train
train: $(TARGETS)
	$(foreach e,$(TARGETS), ./$(e) $x $y  && ) true
	make clean

.PHONY: test
test:
	g++ $(CXXFLAGS) train.cpp -o train1 $(CXXFLAGS7) -lm
	g++ test.cpp -o test $(CXXFLAGS7) -lm
	./train $x $y
	./test $x $y
	make clean
