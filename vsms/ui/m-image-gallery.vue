<template>
  <div>
    <div class="image-gallery" >
        <img v-for="(url,index) in image_urls" :key="index" :src="url" 
        draggable 
        @click="onclick(index)"
        @dragstart="$emit('itemdrag', [$event, index])" 
        :class="['unknown', 'rejected', 'accepted'][ldata.length === 0? 0 : ldata[index].value+1]" />
    </div>
    <m-modal v-if="with_modal" ref='modal2'>
      <img :src="this.image_urls[selection]">
      <!-- <m-annotator :image_url="this.image_urls[selection]" :adata="{boxes:[]}" :read_only="false" /> -->
    </m-modal>
  </div>
</template>
<script>
import MAnnotator from './m-annotator.vue';
import MModal from './m-modal.vue';


 export default {
  components: { 'm-annotator':MAnnotator, 'm-modal':MModal },
  props: {image_urls:{type:Array}, ldata:{type:Array, default:[]}, with_modal:true},
  data : function() { return {selection:null, show_modal:false}},
  created : function (){},
  mounted : function (){
      // this.$refs.gallery_modal.addEventListener('show.bs.modal',this.modalclick);
  },
  methods : {
    onclick(index){
        console.log('click callback')
        this.selection = index;
        // this.$refs.modal2.src = this.image_urls[this.selection];
        if (this.with_modal){
          this.$refs.modal2.active = true; //this.image_urls[this.selection];
        }
        // this.$refs.modal.show()
        this.$emit('update:selection', index); 
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