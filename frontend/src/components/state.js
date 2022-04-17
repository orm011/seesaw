import { reactive } from 'vue'
export const state = reactive({ count: 0 })


function make_counter(init_val){
    return {
        count: init_val,
        increment(){ this.count++}
    }
}

export const counter_list = reactive ({
        counts : [{count:0}, {count:0}, make_counter(1)] 
        
});