#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <limits>
#include <numeric>

using namespace std;


void tokenize(string const &str, const char* delim, vector<int> &out) 
{ 
    char *token = strtok(const_cast<char*>(str.c_str()), delim); 
    while (token != nullptr) 
    { 
        out.push_back(atoi(token)); 
        token = strtok(nullptr, delim); 
    } 
}

// Compute the new quotient and remainder for two subset of processors, and the new local size for each processor
int comp_size(int* new_size, int less_team_size, int local_less_count, int local_great_count, MPI_Comm comm) {
    int p, rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);

    int sq = local_less_count / less_team_size, sr = local_less_count % less_team_size;
    int lq = local_great_count / (p - less_team_size), lr = local_great_count % (p - less_team_size) + less_team_size;
    for (int i = 0; i < p; i++) {
        if (i < less_team_size) new_size[i] = (i < sr)? sq + 1 : sq;
        else new_size[i] = (i < lr)? lq + 1: lq;
    }
    return 0;
}

int comp_counts(int* sendcnts, int* recvcnts, int less_team_size, int* all_less, int* all_great, int* new_size, MPI_Comm comm) {
    int p, rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);

    int j = 0, k = less_team_size; // denote the current receiving processors
    for (int i = 0; i < p; i++) {
        // Fill the small part
        int small_send = all_less[i];
        while (small_send > 0 && j < p) {
            int s = (small_send <= new_size[j])? small_send : new_size[j];
            if (i == rank) sendcnts[j] = s;
            if (j == rank) recvcnts[i] = s;
            small_send -= s;
            new_size[j] -= s;
            // printf("rank:%d, i: %d, j:%d, sendcnts: %d, recvnts:%d\n", rank, i, j, sendcnts[j], recvcnts[i]);
            if (new_size[j] == 0) j++;
        }

        // Fill the large part
        int large_send = all_great[i];
        while (large_send > 0 && k < p) {
            int l = (large_send <= new_size[k])? large_send : new_size[k];
            if (i == rank) sendcnts[k] = l;
            if (k == rank) recvcnts[i] = l;
            large_send -= l;
            new_size[k] -= l;
            if (new_size[k] == 0) k++;
        }
    }
    return 0;
}

int comp_displs(int* sdispls, int* rdispls, int* sendcnts, int* recvcnts, MPI_Comm comm) {
    int p, rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);

    for (int i = 1; i < p; i++) {
        sdispls[i] = sdispls[i - 1] + sendcnts[i - 1];
        rdispls[i] = rdispls[i - 1] + recvcnts[i - 1];
    }
    return 0;
}

void parallel_sort(int* data, int size, MPI_Comm comm){

    int team_size, rank;
    MPI_Comm_size(comm, &team_size);
    MPI_Comm_rank(comm, &rank);

    for (int i = 0; i < size; i ++){
        cout<<data[i]<<" ";
    }
    cout<<endl;

    if (team_size == 1) {
        std::sort(data, data + size);
        return;
    }
    
    // Get the size of the local array, and the size of the global array
    int local_size = size, array_size;
    MPI_Allreduce(&local_size, &array_size, 1, MPI_INT, MPI_SUM, comm);

    // 2. generate a random number across all processors.
    std::srand(std::time(nullptr)); // if we use same seed, all processors will generate the same random number
    int k = std::rand() % array_size; // 0 <= k <= array_size - 1
    // cout << "rank: " << rank << "; random number generated: " << k <<"\n"<< endl;
    // printf("rank: %d, random number generated: %d\n", rank, k);
    // k = 15;
    
    // 3. find which processor has the kth number
    int numFull = array_size % team_size;
    int up = ceil(array_size / (float) team_size);
    int down = floor(array_size / (float) team_size);
    int count = 0;
    int p = 0;

    for(int i = 0; i < team_size; i++){
        int var = (i < numFull) ? up : down;
        count += var;

        if(count > k){
            p = i;
            break;
        }
    }
    // printf("count:%d, local_size:%d\n", count, local_size);
    
    // 4. broadcast pivot to all processors
    int pivot = 0;

    if(rank == p){
        // int previous_count = count - sendcounts[rank];
        int previous_count = count - local_size;
        int index = k - previous_count;
        pivot = data[index];
        // printf("rank %d has the  %d th element.\n The element is %d.\n", rank, k, pivot);
        // cout << "rank: " << rank << " has the " << k << "th element; ";
        // cout << "The element is " << pivot << endl;
    }

    MPI_Bcast(&pivot, 1, MPI_INT, p, comm);
    printf("rank %d and pivot is %d \n", rank, pivot);


    // 5. paritional local array into two subarrays
    vector<int> local_less;
    vector<int> local_great;
    // for(int i=0; i < sendcounts[rank]; i++){
    for(int i=0; i < local_size; i++){
        int ele = data[i];
        if(ele <= pivot){
            local_less.push_back(ele);
        }
        else{
            local_great.push_back(ele);
        }
    }
    int local_less_count = local_less.size();
    int local_great_count = local_great.size();

    // 6. Use all gather to transfer data
    vector<int> all_less; all_less.resize(team_size, -1);
    vector<int> all_great; all_great.resize(team_size, -1);
    vector<int> cur_size(team_size);
    MPI_Allgather(&local_less_count, 1, MPI_INT, &all_less[0], 1, MPI_INT, comm);
    MPI_Allgather(&local_great_count, 1, MPI_INT, &all_great[0], 1, MPI_INT, comm);
    MPI_Allgather(&local_size, 1, MPI_INT, &cur_size[0], 1, MPI_INT, comm);

    int all_less_count = 0;
    int all_great_count = 0;
    for(int i=0; i < team_size; i++){
        all_less_count += all_less.at(i);
        all_great_count += all_great.at(i);
    }
    printf("rank %d with pivot %d and all_less_count %d and all_great_count %d \n", rank, pivot, all_less_count, all_great_count);
    
    // 7. Calculate how to assign processors to the two sub problems
    int less_team_size = std::round( (float) all_less_count / (all_less_count + all_great_count) * team_size);
    less_team_size = less_team_size > 0 ? less_team_size : 1;
    int great_team_size = team_size - less_team_size;
    if(rank == 0){
        printf("less_team_size is %d, great_team_size is %d, all_less is %d, all_great is %d \n", less_team_size, great_team_size, all_less_count, all_great_count);
    }

    // 8. Calculate where to send local_less and local_great
    vector<int> new_size(team_size, 0);
    comp_size(&new_size[0], less_team_size, local_less_count, local_great_count, comm);

    // Compute the send and receive counts
    vector<int> reference(new_size), sendcnts(team_size, 0), recvcnts(team_size, 0);
    comp_counts(&sendcnts[0], &recvcnts[0], less_team_size, &all_less[0], &all_great[0], &reference[0], comm);

    // Compute the send and receive displacement
    vector<int> sdispls(team_size, 0), rdispls(team_size, 0);
    comp_displs(&sdispls[0], &rdispls[0], &sendcnts[0], &recvcnts[0], comm);

    // All-to-all communication
    int new_local_size = new_size[rank];
    int *rbuf = new int[new_local_size];
    MPI_Alltoallv(data, &sendcnts[0], &sdispls[0], MPI_INTEGER, rbuf, &recvcnts[0], &rdispls[0], MPI_INTEGER, comm);

    // 9. Split the communicators into 2 and recursive
    int color = rank < less_team_size ? 0 : 1;
    MPI_Comm new_comm;
    MPI_Comm_split(comm, color, rank, &new_comm);
    parallel_sort(rbuf, new_local_size, new_comm);
    MPI_Comm_free(&new_comm);

    //10. Transfer the data using All-to-all communication
    vector<int> sendcnts_temp(team_size, 0), recvcnts_temp(team_size, 0), sdispls_temp(team_size, 0), rdispls_temp(team_size, 0);
    vector<int> reference_temp(cur_size);
    for (int i = 0, j = 0; i < team_size; i++) {
        int send = new_size[i];
        while (send > 0 && j < team_size) {
            int s = (send <= reference_temp[j])? send : reference_temp[j];
            if (i == rank) sendcnts_temp[j] = s;
            if (j == rank) recvcnts_temp[i] = s;
            send -= s;
            reference_temp[j] -= s;
            if (reference_temp[j] == 0) j++;
        }
    }
    comp_displs(&sdispls_temp[0], &rdispls_temp[0], &sendcnts_temp[0], &recvcnts_temp[0], comm);

    // All-to-to communication
    MPI_Alltoallv(rbuf, &sendcnts_temp[0], &sdispls_temp[0], MPI_INTEGER, data, &recvcnts_temp[0], &rdispls_temp[0], MPI_INTEGER, comm);

    delete [] rbuf;

}


int main(int argc, char* argv[]){
    std::string usage = "usage: ./pqsort [input.txt] [output.txt]";
    if(argc < 3){
        std::cout << usage << std::endl;
        return 0;
    }

    char* input_file_name = argv[1];
    char* output_file_name = argv[2];

    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, team_size;
    MPI_Comm_size(comm, &team_size);
    MPI_Comm_rank(comm, &rank);

    vector<int> numbers;
    int array_size;
    int *sendcounts; 
    int *displs;
    int sum = 0;
    int *data;

    if(rank == 0){
        // -1. Read input file
        int count = 0;
        
        ifstream infile(input_file_name);
        string line;
        while (getline(infile, line))
        {
            if(count > 0){
                const char* delim = " "; 
                tokenize(line, delim, numbers);
            }
            else{
                array_size = atoi(line.c_str());
            }
            count += 1;
        }
        infile.close();

        data = (int*) malloc(sizeof(int)*array_size);
        // vector<int> data;
        for(int i=0; i < array_size; i++){
            data[i] = numbers.at(i);
            cout << data[i] << " ";
        }
        cout << endl;
    }

    // 0. Broadcase array size to all processors
    MPI_Bcast(&array_size, 1, MPI_INT, 0, comm);

    sendcounts = (int*) malloc(sizeof(int)*team_size);
    displs = (int*) malloc(sizeof(int)*team_size);

    int N = array_size; 
    int rem = N % team_size; // 
    for (int i = 0; i<team_size; i++) {
        sendcounts[i] = N / team_size;
        if (rem-- > 0)
            sendcounts[i]++;
        //an array of the displacement (in terms of array indices) of each process's slice of the data array.
        displs[i] = sum;
        sum += sendcounts[i];
    }


    // 1. Block distribute array to all processors
    // int rec_buf[sendcounts[rank]];
    // MPI_Scatterv(data, sendcounts, displs, MPI_INT, &rec_buf, sendcounts[rank], MPI_INT, 0, MPI_COMM_WORLD);
    vector<int> rec_buf;
    rec_buf.resize(sendcounts[rank]);
    MPI_Scatterv(data, sendcounts, displs, MPI_INT, &rec_buf[0], sendcounts[rank], MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier (MPI_COMM_WORLD);
    // start timer
    double t_start, t_end;
    t_start = MPI_Wtime();

    parallel_sort(&rec_buf[0], rec_buf.size(), MPI_COMM_WORLD);

    MPI_Barrier (MPI_COMM_WORLD);
    // get elapsed time in seconds
    t_end = MPI_Wtime();
    double time_secs = (t_end - t_start);

    // output time
    if (rank == 0) {
        cout<< "sorting took: " << time_secs << " s"<<endl;
    }

    // get local size
    int local_size = rec_buf.size();
    vector<int> result;

    // master process receive results
    if (rank == 0)
    {
        // gather local array sizes
        vector<int> local_sizes(team_size);
        MPI_Gather(&local_size, 1, MPI_INT, &local_sizes[0], 1, MPI_INT, 0, comm);

        // gather-v to collect all the elements
        int total_size = accumulate(local_sizes.begin(), local_sizes.end(), 0);
        result.resize(total_size);

        // get receive displacements
        vector<int> displs(team_size, 0);
        for (int i = 1; i < team_size; ++i)
            displs[i] = displs[i - 1] + local_sizes[i - 1];

        // gather v the vector data to the root
        MPI_Gatherv(&rec_buf[0], local_size, MPI_INT,&result[0], &local_sizes[0], &displs[0], MPI_INT, 0, comm);
    }
    // else: send results
    else
    {
        MPI_Gather(&local_size, 1, MPI_INT, NULL, 1, MPI_INT, 0, comm);
        MPI_Gatherv(&rec_buf[0], local_size, MPI_INT,NULL, NULL, NULL, MPI_INT, 0, comm);
    }

    for (unsigned int i = 0; i < result.size(); ++i){
        cout << result[i] << endl;
    }

    MPI_Finalize();
    return 0;

}

