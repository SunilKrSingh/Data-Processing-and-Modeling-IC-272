#include<iostream>
#include<climits>
using namespace std;

void swap(int *a, int *b)
{
    int t = *a;
    *a = *b;
    *b = t;
}

class min_bin_heap
{
    
    int length;
    int heapsize;
    int *A;
    public:
        min_bin_heap()
        {
            length = 0;
            heapsize = 0;
            A = nullptr;
        };
        min_bin_heap(int a[], int N, int n)
        {
            A = a;
            length = N;
            heapsize = n;
        };
        int parent(int i)
        {
            return (i-1)/2;
        }
        int left(int i)
        {
            return 2*i+1;
        }
        int right(int i)
        {
            return 2*i+2;
        }
        void min_heapify(int i)
        {
            int l = left(i);
            int r = right(i);
            int smallest;
            if (l<heapsize && A[l] < A[i])
            {
                smallest = l;
            }
            else
            {
                smallest = i;
            }
            if (r < heapsize && A[r]<A[smallest])
            {
                smallest = r;
            }
            if (smallest != i)
            {
                swap(&A[i], &A[smallest]);
                min_heapify(smallest);
            }
        }
        void build_min_heap()
        {
            int k = length/2;
            for (k; k>=0; k--)
            {
                min_heapify(k);
            }
        }
        
        
        /*
        int min_heap()
        {
            return A[0];
        }

        int min_heap_extract_min()
        {
            if (heapsize < 1)
            {
                cout<<"heapsize underflow"<<endl;
            }
            else
            {
                int min = A[0];
                A[0] = A[heapsize - 1];
                heapsize -= 1;
                min_heapify(0);
                return min;
            }
        }
        */
        void min_heap_dec_key(int i, int key)
        {
            if (key > A[i])
            {
                cout<<"new key is greater than current key"<<endl;;
            }
            else
            {
                A[i] = key;
                while(i>0 && A[parent(i) > A[i]])
                { 
                    swap(&A[i], &A[parent(i)]);
                    i = parent(i);
                }
            }
            
        }
        /*
        void min_heap_insert_key(int x)
        {
            if (heapsize == length)
            {
                cout<<"error : heapsize size is full, can't insert element"<<endl;
            }
            else
            {
                heapsize += 1;
                A[heapsize-1] = x;
                int z = heapsize - 1;
                min_heap_dec_key(z, x);
            } 
            build_min_heap();                    
        }
        */
        void print()
        {
            for(int i=0; i<heapsize; i++)
            {
                cout<<A[i]<<" ";
            }
        }

        int get_max()
        {
            int i = 0;
            int t = 0;
            while(i<heapsize && t<=A[i])
            {
                t = A[i];
                i++;
            }
            min_heap_dec_key(i-1,A[i-1]/2);
            return t;

        }
};

int main()
{
    int N;
    int n;
    
    
    cout<<"enter the maximum size of heapsize : "<<endl;
    cin>>N;
    cout<<"enter the current size of heapsize :"<<endl;
    cin>>n;
    cout<<"enter the "<<n<<" elements for array : "<<endl;
    int a[n];
    for(int i=0; i<n; i++)
    {
        cin>>a[i];
    }
    cout<<endl;
    min_bin_heap bh(a, N, n);
    bh.build_min_heap();
    bh.print();
    cout<<endl;
    cout<<"enter the of chocolates kid can eat: "<<endl;
    int k_kid;
    cin>>k_kid;
    int sum=0;
    for(int i=0; i<k_kid; i++)
    {
        sum = sum + bh.get_max();

    }
    cout<<"maximum number of chocolates eaten : "<<sum<<endl;
    return 0;
}