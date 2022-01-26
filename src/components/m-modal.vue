<template>
  <!-- mostly based on https://www.w3schools.com/howto/howto_css_modal_images.asp -->
  <div
    :class="`my-modal ${active ? 'my-modal-active': ''}` "
  >
    <span
      class="close"
      @click="close"
    >&times;</span>
    <div
      class="my-modal-content"
    >
      <slot />
    </div>
  </div>
</template>
<script>
export default {
    name : 'MModal',
    data : function(){return {active:true, handler:null}},
    emits : ['arrow', 'close'],
    mounted : function () {
      this.$data.handler = this.handle_keyup;
      console.log('added handler', this)
      window.addEventListener('keyup', this.handle_keyup)
    },
    unmounted : function (){
      console.log('removed handler')
      window.removeEventListener('keyup', this.handle_keyup);
    },
    methods : {

        close(){
            this.$emit('close');
            // this.active = false;
            // this.modal.hide()
        },
        handle_keyup(ev){
            console.log('within handler', this, ev)
            if (ev.code === 'ArrowLeft' || ev.code === 'ArrowRight'){
              this.$emit('arrow', ev.code); 
            } else if (ev.code == 'Escape') {
              this.$emit('close');
            }
        },
        show(){
            // this.active = true;
            // this.modal.show()
        }
    }
}
</script>
<style scoped>
/* The Modal (background) */
.my-modal {
  display: none; /* Hidden by default */
  position: fixed; /* Stay in place */
  text-align: center;
  z-index: 1021; /* Sit on top of left menu, which is at 100, and top bar, which is at 1020 */
  padding-top: 100px; /* Location of the box */
  left: 0;
  top: 0;
  width: 100%; /* Full width */
  height: 100%; /* Full height */
  overflow: auto; /* Enable scroll if needed */
  background-color: rgb(0,0,0); /* Fallback color */
  background-color: rgba(0,0,0,0.9); /* Black w/ opacity */
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

/* 100% Image Width on Smaller Screens */
@media only screen and (max-width: 700px){
  .modal-content {
    width: 100%;
  }
}
</style>