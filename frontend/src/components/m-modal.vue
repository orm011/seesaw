<template>
  <!-- mostly based on https://www.w3schools.com/howto/howto_css_modal_images.asp -->
  <div
    :class="`my-modal ${active ? 'my-modal-active': ''}` "
  >
    <!-- <span
      class="close"
      @click="close"
    >&times;</span> -->
    <div  
      class="my-modal-content"
    >
      <slot />
    </div>
    <div class="keyword-text">
      <span> 'Esc' for going back to main view </span>
      <span> 'Left' and 'Right' arrow for previous/next image</span>
      <!-- <span> 'Space' to show fine-grained activation </span> -->
    </div>
  </div>
</template>
<script lang="ts">

import {defineComponent} from 'vue';

export default defineComponent({
    name : 'MModal',
    data : function(){return {active:true, handler:null}},
    emits : ['modalKeyUp'],
    mounted : function () {
      this.$data.handler = this.handle_keyup;
      window.addEventListener('keyup', this.handle_keyup)
      let elt = document.getElementsByTagName('body')[0];
      elt.classList.add("modal-is-active");
    },
    unmounted : function (){
      window.removeEventListener('keyup', this.handle_keyup);
      let elt = document.getElementsByTagName('body')[0]
      elt.classList.remove("modal-is-active");
    },
    methods : {
        handle_keyup(ev){
            this.$emit('modalKeyUp', ev);
        },
    }
})
</script>
<style>
/* The Modal (background) */
.my-modal {
  display: none; /* Hidden by default */
  position: fixed; /* Stay in place */
  text-align: center;
  z-index: 1021; /* Sit on top of left menu, which is at 100, and top bar, which is at 1020 */
  padding-top: 10px; 
  left: 0;
  top: 0;
  width: 100%; /* Full width */
  height: 100%; /* Full height */
  overflow: auto; /* Enable scroll if needed */
  background-color: rgb(0,0,0); /* Fallback color */
  background-color: rgba(0,0,0,0.9); /* Black w/ opacity */
}

/* disables scrolling of background while modal is open 
including if we press space bar*/
body.modal-is-active {
  height: 100vh; 
  overflow: hidden;
} 

.my-modal-active {
  display: block;
}


/* Modal Content (image) */
.my-modal-content {
  margin: auto;
  display: block;
  width: fit-content;
  height: fit-content;
  /* max-width: 700px; */
}

/* Caption of Modal Image */
/* #caption {
  margin: auto;
  display: block;
  width: 80%;
  max-width: 700px;
  text-align: center;
  color: #ccc;
  padding: 10px 0;
  height: 150px;
} */

/* Add Animation */
.modal-content {  
  -webkit-animation-name: zoom;
  -webkit-animation-duration: 0.6s;
  animation-name: zoom;
  animation-duration: 0.6s;
}

@-webkit-keyframes zoom {
  from {-webkit-transform:scale(0)} 
  to {-webkit-transform:scale(1)}
}

@keyframes zoom {
  from {transform:scale(0)} 
  to {transform:scale(1)}
}

/* The Close Button */
.close {
  position: absolute;
  top: 15px;
  right: 35px;
  color: #f1f1f1;
  font-size: 40px;
  font-weight: bold;
  transition: 0.3s;
}

.close:hover,
.close:focus {
  color: #bbb;
  text-decoration: none;
  cursor: pointer;
}

.keyword-text > span {
  color: rgb(240,240,240);
  font-size: 15px;
  display:block;
}

/* 100% Image Width on Smaller Screens */
@media only screen and (max-width: 700px){
  .modal-content {
    width: 100%;
  }
}
</style>