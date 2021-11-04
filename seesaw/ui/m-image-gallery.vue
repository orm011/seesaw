<template>
  <div>
    <div class='row'>
    <div class="image-gallery" >
        <img v-for="(data,index) in imdata" :key="index" :src="data.url" 
        draggable 
        @click="onclick(index)"
        @dragstart="$emit('itemdrag', [$event, index])" 
        :class="get_class(index)" />
    </div>
    </div>
    <!-- <div class='row'> -->
            <!-- <img :src="this.imdata[selection].url">  -->
    <!-- </div> --> 
    <m-modal v-if="with_modal" ref='modal2' @keyup.esc='this.$refs.modal2.close()'  tabindex='0' >
        <m-annotator  ref='annotator' :imdata="this.imdata[selection]" :read_only="false" 
            v-on:esc='this.$refs.modal2.close()' @keyup.esc='this.$refs.modal2.close()'   tabindex='1' />
        <!-- <button class="navbar-toggler position-absolute d-md-none collapsed" type="button" 
          data-bs-toggle="collapse" data-bs-target="#sidebarMenu" aria-controls="sidebarMenu" 
            aria-expanded="false" aria-label="Toggle navigation">
              <span class="navbar-toggler-icon"></span>
        </button> -->
        <button class="btn btn-dark bton-block" @click="copyref"> 
            Fill with reference
        </button>
    </m-modal>
  </div>
</template>
<script>
import MAnnotator from './m-annotator.vue';
import MModal from './m-modal.vue';


 export default {
  components: { 'm-annotator':MAnnotator, 'm-modal':MModal },
  props: { imdata:{type:Array, default:[]}, with_modal:true},
  data : function() { return {selection:null, show_modal:false}},
  created : function (){},
  mounted : function (){
      // this.$refs.gallery_modal.addEventListener('show.bs.modal',this.modalclick);
  },
  methods : {
    // kinds of feedback: 
    // 1. this image does not have what I'm looking for (checkbox?)
    // 2. 

    get_class(index){
      let ldata = this.imdata[index];
      if (ldata.boxes == null){
        return 'unknown'
      } else if (ldata.boxes.length > 0) {
        return 'accepted'
      } else if (ldata.boxes.length === 0){
        return 'rejected'
      }
    },
    onclick(index){
        console.log('click callback')
        this.selection = index;
        if (this.with_modal){
          this.$refs.modal2.active = true;
        }
        // this.$refs.modal.show()
        this.$emit('update:selection', index); 
    },
    copyref(){
        console.log('click copyref');
        const reflabels = this.imdata[this.selection].refboxes;
        let adata = this.imdata[this.selection].boxes
        console.log(reflabels)
        for (const obj of reflabels){
                adata.push(obj)  
        }

        this.$refs.annotator.load_current_box_data()
    }
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