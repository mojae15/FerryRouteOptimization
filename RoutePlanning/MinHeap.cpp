#include "MinHeap.hpp"
#include <climits>
#include <iostream>

void swap(Vertex* x, Vertex* y){
    Vertex temp = *x;
    *x = *y;
    *y = temp;
}

Vertex MinHeap::extractMin(){

    if (cur_size <= 0){
        //Do some error here
    }
    if (cur_size == 1){
        cur_size--;
        return heap[0];
    }

    Vertex root = heap[0];
    heap[0] = heap[cur_size - 1];
    cur_size--;
    minHeapify(0);
    return root;

}

void MinHeap::minHeapify(int i){

    int l = left(i);
    int r = right(i);
    int smallest = i;
    if (l < cur_size && heap[l].cost < heap[i].cost){
        smallest = l;
    } 
    if (r < cur_size && heap[r].cost < heap[smallest].cost){
        smallest = r;
    }
    if (smallest != i){
        swap(&heap[i], &heap[smallest]);
        minHeapify(smallest);
    }



}

void MinHeap::decreaseKey(int index, double new_val){


    heap[index].cost = new_val;

    
    while ( index != 0 && heap[parent(index)].cost > heap[index].cost){
        swap(&heap[index], &heap[parent(index)]);
        index = parent(index);
    }

    

}

// void MinHeap::deleteKey(int i){

//     decreaseKey(i, INT_MIN);
//     extractMin();

// }

void MinHeap::insertKey(Vertex i ){

    if (cur_size == capacity){
        
        //don't know homie
        std::cout << "Too many elements" << std::endl;
        return;
    }

    cur_size++;
    int j = cur_size -1;
    heap[j] = i;
    
    while ( j != 0 && heap[parent(j)].cost > heap[j].cost){
        swap(&heap[j], &heap[parent(j)]);
        j = parent(j);
    }

}


