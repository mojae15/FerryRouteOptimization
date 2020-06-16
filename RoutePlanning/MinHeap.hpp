#ifndef MINHEAP_HPP
#define MINHEAP_HPP

#include "Vertex.hpp"
#include <iostream>

class MinHeap {

    private:
        int capacity;
        int cur_size;
        Vertex *heap;

    public:
        MinHeap(int cap){
            capacity = cap;
            cur_size = 0;
            heap = new Vertex[cap];
        }

        Vertex extractMin();

        void minHeapify(int i);

        int parent(int i ) {
            return (i-1)/2;
        }

        int left(int i ){
            return (2*i + 1);
        }

        int right(int i ){
            return (2*i + 2);
        }

        void decreaseKey(int index, double new_val);

        Vertex getMin(){
            return heap[0];
        }

        // void deleteKey(int i );

        void insertKey(Vertex i);

        int getSize(){
            return cur_size;
        }

};

#endif