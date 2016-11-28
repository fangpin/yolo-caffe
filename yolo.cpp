#include <string>
#include <vector>
#include <iostream>
#include <cassert>
#include <ctime>
#include <cstdio>
#include <vector>
#include <cmath>
#include <algorithm>
#include <sys/time.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "boost/algorithm/string.hpp"
#include "google/protobuf/text_format.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/data_transformer.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"

using caffe::Blob;
using caffe::Caffe;
using caffe::Datum;
using caffe::Net;
using caffe::TransformationParameter;
using caffe::DataTransformer;

using std::string;
using std::vector;
using std::cout;
using std::cerr;
using std::endl;



#define DEBUG 1

struct Box{
    float x,y,w,h;
};

struct Node{
    int index,kind;
};

class Yolo{
public:
    Yolo(const std::string& modelFile, const std::string& weightFile, bool gpu=true ,float pthreshold=0.2, float piouThreshold=0.5, int pnumClass=20, int pnumBox=2, int pgrideSize=7){
        threshold = pthreshold;
        iouThreshold = piouThreshold;
        numClass = pnumClass;
        numBox = pnumBox;
        grideSize = pgrideSize;
        useGpu = gpu;

        if(gpu){
            Caffe::set_mode(Caffe::GPU);
        }
        else{
            Caffe::set_mode(Caffe::CPU);
        }
        net.reset(new Net<float>(modelFile, caffe::TEST));
        net->CopyTrainedLayersFrom(weightFile);
        CHECK_EQ(net->num_inputs(), 1) << "Network should have exactly one input.";
        CHECK_EQ(net->num_outputs(), 1) << "Network should have exactly one output.";
        inputBlob = net->input_blobs()[0];
        imgChannels = inputBlob->channels();
        imgSize = cv::Size(inputBlob->width(), inputBlob->height());
    }

    void predict(cv::Mat& img){
        cv::Mat imgResized;
        if(img.size() != imgSize){
            cv::resize(img, imgResized, imgSize);
        }
        else{
            imgResized = img;
        }
        vector<Box> boxes(grideSize*grideSize*numBox);
        cmp.probs.resize(grideSize*grideSize*numBox);
        for(int i=0; i<grideSize * grideSize * numBox; ++i)
            cmp.probs[i].resize(numClass);
        wrapInputBlob(imgResized);
        vector<float> result = forward();
        parse(result, img.size(), boxes, cmp.probs);
        doSort(boxes, cmp.probs);
        draw(img, boxes, cmp.probs);
        // return img;
    }


private:
    vector<float> forward(){
        vector<float> output;
        const float* blobData;

        #ifdef DEBUG
        struct timeval t_start,t_end;
        gettimeofday(&t_start, NULL);
        #endif

        net->Forward();

        #ifdef DEBUG
        gettimeofday(&t_end, NULL);
        int timeuse = 1000000 * ( t_end.tv_sec - t_start.tv_sec ) + t_end.tv_usec - t_start.tv_usec;
        std::cerr<<"time: "<<timeuse<<" us\n"<<std::endl;
        #endif

        boost::shared_ptr<Blob<float> > outputBlob = net->blob_by_name("result");
        int dims = outputBlob->count();
        if(useGpu)
            blobData = outputBlob->gpu_data();
        else
            blobData = outputBlob->cpu_data();
        copy(blobData, blobData+dims, back_inserter(output));
        return output;
    }

    void wrapInputBlob(const cv::Mat& imgResized){
        const float scale = 0.0039215684;
        TransformationParameter param;
        param.set_scale(scale);
        DataTransformer<float> dt(param, caffe::TEST);
        dt.Transform(imgResized, inputBlob);
    }

    void parse(const vector<float>& result, const cv::Size& size, vector<Box>& boxes, vector<vector<float> >& probs){
        // grideSize * grideSize * numClass + grideSize * grideSize * confidence + grideSize * grideSize * (numBox * 4)
        // default   7*7*20 + 7*7*1 + 7*7*2*4
        // 4 means (x, y, width, height)
        for(int i=0; i<grideSize*grideSize; ++i){
            int row = i / grideSize;
            int col = i % grideSize;
            for(int j=0; j<numBox; ++j){
                int index = i*numBox + j;
                int pIndex = grideSize * grideSize * numClass + index;
                int boxIndex = grideSize * grideSize * (numClass + numBox) + (index << 2);
                float scale = result[pIndex];
                boxes[index].x = (result[boxIndex + 0] + col) / grideSize;
                boxes[index].y = (result[boxIndex + 1] + row) / grideSize;
                boxes[index].w = pow(result[boxIndex + 2], 2);
                boxes[index].h = pow(result[boxIndex + 3], 2);

                for(int k=0; k<numClass; ++k){
                    int classIndex = i*numClass;
                    float prob = scale*result[classIndex+k];
                    probs[index][k] = (prob > threshold) ? prob:0;
                }
            }
        }
    }

    void doSort(vector<Box>& boxes, vector<vector<float> >& probs){
        int total = grideSize * grideSize * numBox;
        vector<Node> vec(total);
        for(int i=0; i<total; ++i){
            vec[i].index = i;
            vec[i].kind = 0;
        }
        for(int i=0; i<numClass; ++i){
            for(int j=0; j<total; ++j){
                vec[j].kind = i;
            }
            sort(vec.begin(), vec.end(), cmp);
            for(int j=0; j<total; ++j){
                if(!probs[vec[j].index][i]) continue;
                Box &a = boxes[vec[j].index];
                for(int k=j+1; k<total; ++k){
                    Box &b = boxes[vec[k].index];
                    if(boxIou(a, b) > iouThreshold)
                        probs[vec[k].index][i] = 0;
                }
            }
        }
    }

    void draw(cv::Mat& img, vector<Box>& boxes, vector<vector<float> >& probs){
        static const string names[] = {"airplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"};
        int num = grideSize * grideSize * numBox;
        int wImg = img.cols;
        int hImg = img.rows;
        for(int i = 0; i < num; ++i){
            int kind = maxIndex(probs[i], numClass);
            float prob = probs[i][kind];
            if(prob > threshold){

                #ifdef DEBUG
                // printf("%ll: %.0f%%\n");
                #endif

                const Box& b = boxes[i];

                int left  = (b.x-b.w/2.)*wImg;
                int right = (b.x+b.w/2.)*wImg;
                int top   = (b.y-b.h/2.)*hImg;
                int bot   = (b.y+b.h/2.)*hImg;

                if(left < 0) left = 0;
                if(right > wImg-1) right = wImg-1;
                if(top < 0) top = 0;
                if(bot > hImg-1) bot = hImg-1;

                cv::Point leftTop = cv::Point(left, top);
                cv::Point rightBottom = cv::Point(right, bot);
                cv::rectangle(img, leftTop, rightBottom, cv::Scalar(255, 0 ,0));
                putText(img, names[kind].c_str(), leftTop, CV_FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 0, 0));
                cout<<names[kind]<<":\t"<<prob<<endl;
            }
        }
    }



    static float boxIou(const Box& a, const Box& b){
        return boxIntersection(a, b) / boxUnion(a,b);
    }

    static float boxIntersection(const Box& a, const Box& b){
        float w = overlap(a.x, a.w, b.x, b.w);
        float h = overlap(a.y, a.h, b.y, b.h);
        if(w<0 || h<0) return 0;
        return w*h;
    }

    static float boxUnion(const Box& a, const Box& b){
        return a.w * a.h + b.w * b.h - boxIntersection(a, b);
    }

    static float overlap(float x1, float w1, float x2, float w2){
        float l1 = x1 - w1/2;
        float l2 = x2 - w2/2;
        float left = l1 > l2 ? l1 : l2;
        float r1 = x1 + w1/2;
        float r2 = x2 + w2/2;
        float right = r1 < r2 ? r1 : r2;
        return right - left;
    }



    static int maxIndex(const vector<float>& a, int n){
        if(n<=0) return -1;
        int i, max_i = 0;
        float maxn = a[0];
        for(i = 1; i < n; ++i){
            // cout<<"a "<<i<<"\t"<<a[i]<<endl;
            if(a[i] > maxn){
                maxn = a[i];
                max_i = i;
            }
        }
        return max_i;
    }


private:
    boost::shared_ptr<Net<float> > net;
    Blob<float>* inputBlob;
    cv::Size imgSize;
    int imgChannels;
    bool useGpu;
    float threshold;
    float iouThreshold;
    int numClass;
    int numBox;
    int grideSize;
    struct Cmp{
        vector<vector<float> > probs;
        bool operator()(const Node& a, const Node& b){
            return probs[a.index][a.kind] > probs[b.index][b.kind];
        }
    }cmp;

    // {"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"};
    // static const enum Classes{
    //     aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor,
    // }classes;
};



int main(int argc,char **argv){
    Yolo yolo("/home/fangpin/caffe-cpu/examples/yolo/yolo_tiny_deploy.prototxt", "/home/fangpin/caffe-cpu/examples/yolo/yolo_tiny.caffemodel", false);
    cout<<argv[1]<<endl;
    cv::Mat mat1;
    if(argc > 1)
        mat1 = cv::imread(argv[1]);
    else
        mat1 = cv::imread("/home/fangpin/caffe-cpu/examples/images/car.jpg");
    yolo.predict(mat1);
    cv::namedWindow("image", cv::WINDOW_AUTOSIZE);
    cv::imshow("image", mat1);
    cv::waitKey();
    // yolo.predict(mat2);
    return 0;
}
