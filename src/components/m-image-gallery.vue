<template>
  <div>
    <div class='row'>
    <div class="image-gallery" >
      <div v-for="(data,index) in initial_imdata" :key="index*10000 + (data.boxes == null ? 0 : data.boxes.length)">
        <!-- img is much more light weight to render, and the common case is no labels -->
        <img v-if="data.boxes == null || data.boxes.length === 0" :src="data.url" @click="onclick(index)" :class="get_class(index)" />
        <m-annotator v-else ref="annotators" :initial_imdata=data :read_only="true" v-on:cclick="onclick(index)" />
      </div>
    </div>
    </div>
    <m-modal v-if="selection != null" ref='modal' v-on:close='close_modal()' tabindex='0' >
      <div class='row'>
        <m-annotator  ref='annotator' :initial_imdata="initial_imdata[selection]" :key="index*10000 + (initial_imdata[selection].boxes == null ? 0 :
         initial_imdata[selection].boxes.length)"  :read_only="false"  tabindex='1' v-on:box-save="box_save(selection, $event)" />
      </div>
        <!-- <button class="navbar-toggler position-absolute d-md-none collapsed" type="button" 
          data-bs-toggle="collapse" data-bs-target="#sidebarMenu" aria-controls="sidebarMenu" 
            aria-expanded="false" aria-label="Toggle navigation">
              <span class="navbar-toggler-icon"></span>
        </button> -->
      <div class='row'>
        <button v-if="refmode" class="btn btn-dark bton-block" @click="$emit('copy-ref', selection)"> 
            Autofill ({{initial_imdata[selection].refboxes.length}} boxes)
        </button>
        <!-- <button v-if="refmode" class="btn btn-dark bton-block" @click="copyref(selection)"> 
            Mark not-relevant
        </button> -->
      </div>
    </m-modal>
  </div>
</template>
<script>
import MAnnotator from './m-annotator.vue';
import MModal from './m-modal.vue';


 export default {
  name : 'm-image-gallery',
  components: { 'm-annotator':MAnnotator, 'm-modal':MModal },
  props: { initial_imdata:{type:Array, default: () => []}, refmode:Boolean },
  data : function() { return { selection:null }},
  created : function (){},
  mounted : function (){
      // this.$refs.gallery_modal.addEventListener('show.bs.modal',this.modalclick);
  },
  methods : {
    // kinds of feedback: 
    // 1. this image does not have what I'm looking for (checkbox?)
    // 2. 
    get_class(index){
      
      let ldata = this.initial_imdata[index];
      if (ldata.boxes == null){
        return 'unknown'
      } else if (ldata.boxes.length > 0) {
        return 'accepted'
      } else if (ldata.boxes.length === 0){
        return 'rejected'
      }
    },
    close_modal(){ // closing modal emits a data edit event
      // this.box_save(index);
      this.$refs.annotator.save_current_box_data();
      this.selection  = null;
    },
    box_save(index, boxlist){
      // let imdict = this.initial_imdata[index];
      console.log('(gallery) saving boxes...', index, boxlist)
      this.$emit('data_update', {idx:index, boxes:boxlist})
    },
    onclick(index){
      this.selection = index;
    },
    // copyref(index){
    //     let imdict = this.imdata[index];

    //     console.log('click copyref');
    //     const reflabels = imdict.refboxes;

    //     if (imdict.boxes == null){
    //       imdict.boxes = []
    //     }

    //     for (const obj of reflabels){
    //       imdict.boxes.push(obj)  
    //     }

    //     this.$refs.annotator.load_current_box_data()
    // }
  }
}
</script>
<style scoped>
/* makes a grid https://css-tricks.com/seamless-responsive-photo-grid/
  with no whitespace gaps... not quite clear to me how it picks layout */

.image-gallery {
  /* Prevent vertical gaps */
  display: flex;
  flex-wrap: wrap;
  line-height: 0;
  row-gap: 0px;
  justify-content: center;
  /* column-count:         5;*/
  column-gap: 0px;
}

.image-gallery img {
  /* Just in case there are inline attributes */
  /* width: auto !important;
  height: 300px !important; */
  width: auto !important;
  height: auto !important;
  max-height: 200px;
  /*max-width: 400px; */
  object-fit: scale-down; /* never rescale up */ 
  /* flex-grow: 2; */
  /* flex-shrink: 2; */
  /* box-shadow: 0 1px 2px rgba(0,0,0,.15); */
  /* transition: all 100ms ease-out; */
  transition: opacity 100ms;
}
.image-gallery img.rejected {
  opacity: .5;
  border: 5px solid red;
}

.image-gallery .rejected:hover {
  opacity: 1;
}

.image-gallery .accepted {
  border: 5px solid green;
  opacity: .5;
}

.image-gallery .accepted:hover {
  opacity: 1;
}

.image-gallery .unknown:hover {
  /* box-shadow: 0 3px 5px rgba(0,0,0,.3); */
  opacity: .5;
  /* translateY(.5%) scale(1.01, 1.01); */
  /* translate bc top image wraps otherwise */ 
}

</style>