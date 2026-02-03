#pragma once
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <limits>
using namespace std;

class GMMModel{
public:
    GMMModel(int H, int W, int K, float alpha, float threshold, float bg_threshold, 
             const float* first_frame_ptr)
        : H_(H), W_(W), K_(K),
        alpha_(alpha), threshold_(threshold),bg_threshold_(bg_threshold)
    {
        const int HW = H_ * W_;
        means_.assign(K_ * HW, 0.0f);
        vars_.assign(K_ * HW, 0.01f);
        weights_.assign(K_ * HW, 0.0f);

        // init: first Gaussian gets first frame, weight = 1
        for(int i = 0 ; i < HW ; i++){
            means_[0*HW + i] = first_frame_ptr[i];
            weights_[0*HW + i] = 1.0f;
        }
    }

    // Apply one frame -> output mask (0 = BG, 255 = FG)
    void apply(const float* frame_ptr, uint8_t* out_mask_ptr){
        const int HW = H_ * W_;
        const float min_var = 1e-4f;

        // per-pixel arrays
        // best_k: index of best matching gaussian (if any)
        // any_match: whether any gaussian matched
        vector<int> best_k(HW,0);
        vector<uint8_t> any_match(HW,0);
        
        //-------------------------------------
        // step 1: best match per pixel
        //-------------------------------------
        for(int idx = 0; idx < HW; idx++){
            float x = frame_ptr[idx];
            
            float best_dist = numeric_limits<float>::max();
            int best = -1;
            bool matched_any = false;

            for(int k = 0; k < K_; k++){
                int off = k * HW + idx;
                float var = max(vars_[off], min_var);
                float stdv = sqrt(var);

                float dist = fabs(x - means_[off]) / stdv;
                
                if(dist < threshold_){
                    matched_any = true;
                    if(dist < best_dist){
                        best_dist = dist;
                        best = k;
                    }
                }
            }

            any_match[idx] = matched_any ? 1 : 0;
            best_k[idx] = best;

        }

        // ---------------------------------------
        // Step 2: background decision (0/255 mask)
        // ---------------------------------------
        // For each pixel, rank gaussians by fitness = weight/std
        // Background set = top B gaussians until cumulative weight >= bg_threshold

        for(int idx = 0 ; idx < HW ; idx ++){
            if(!any_match[idx]){
                out_mask_ptr[idx] = 255 ;           // foreground if no gaussian matached
                continue;
            }

            // build ranking for this pixel
            vector<int> order(K_);
            for(int k = 0 ; k < K_ ; k++){
                order[k] = k;
            }

            auto fitness = [&](int k){
                int off = k * HW + idx;
                float var = max(vars_[off], min_var);
                float stdv = sqrt(var);

                return weights_[off] / stdv;
            };
            
            sort(order.begin(), order.end(), [&](int a, int b) {
                return fitness(a) > fitness(b);
            });
            
            // find the best_k is inside background set
            float cum = 0.0f;
            bool is_bg = false;

            for(int r = 0; r < K_; r++){
                int k = order[r];
                int off = k * HW + idx;
                
                // background set includes this gaussian if cum weight BEFORE it < threshold
                if(cum < bg_threshold_){
                    if(k == best_k[idx]){
                        is_bg = true;
                    }
                }

                cum += weights_[off];
                if(cum >= bg_threshold_) break;
            }

            out_mask_ptr[idx] = is_bg ? 0 : 255;
        }
        
        // ---------------------------
        // Step 3: update model
        // ---------------------------
        // Canonical update:
        // 1) decay all weights
        // 2) update matched gaussian (mean/var/weight)
        // 3) if no match -> replace weakest gaussian
        // 4) normalize weights
        //
        // NOTE: for parity with your current python vec,
        // we update all pixels (both bg and fg).
        // Later you can gate updates using out_mask_ptr[idx]==0.

        // 3.1 decay weights
        for (int k = 0; k < K_; k++) {
            int base = k * HW;
            for (int idx = 0; idx < HW; idx++) {
                weights_[base + idx] *= (1.0f - alpha_);
            }
        }

        // 3.2 update matched / replace unmatched
        for (int idx = 0; idx < HW; idx++) {
            float x = frame_ptr[idx];

            if (any_match[idx]) {
                int k = best_k[idx];
                int off = k * HW + idx;
                
                // mean update
                float diff = x - means_[off];
                means_[off] += alpha_ * diff;

                // var update: EMA
                vars_[off] += alpha_ * (diff * diff - vars_[off]);

                // reward matched weight
                weights_[off] += alpha_;
            } else {
                // replace weakest gaussian for this pixel
                int weakest = 0;
                float wmin = numeric_limits<float> :: max();
                for (int k = 0; k < K_; k++) {
                    int off = k * HW + idx;
                    if (weights_[off] < wmin) {
                        wmin = weights_[off];
                        weakest = k;
                    }
                }

                int off = weakest * HW + idx;
                means_[off] = x;
                vars_[off] = 0.02f;     // keep same as your python vec parity
                weights_[off] = 0.05f;  // small initial weight
            }
        }

        // 3.3 normalize weights per pixel
        for (int idx = 0; idx < HW; idx++) {
            float sumw = 0.0f;
            for (int k = 0; k < K_; k++) {
                sumw += weights_[k * HW + idx];
            }
            sumw = sumw + 1e-8f;

            for (int k = 0; k < K_; k++) {
                weights_[k * HW + idx] /= sumw;
            }
        }
    }

private:
    int H_, W_, K_;
    float alpha_, threshold_, bg_threshold_;

    // stored as [k][H][W] flattened into 1D
    vector<float> means_;
    vector<float> vars_;
    vector<float> weights_;
    
};