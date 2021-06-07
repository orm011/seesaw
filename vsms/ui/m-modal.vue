<template>
    <div class="modal fade" ref="modal">
        <div class="modal-dialog modal-xl">
            <div class="modal-content">
                <!-- <div class="modal-header">
                    <h4 class="modal-title">
                        Title
                    </h4>
                </div> -->
                <div class="modal-body">
                        <slot>
                        </slot>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-default" @click="close">Close</button>
                    <!-- <button type="button" class="btn btn-primary">Save changes</button> -->
                </div>
            </div>
        </div>
    </div>
</template>
<script>
export default {
    data : function(){return {modal:null}},
    mounted : function () {
        this.modal = new bootstrap.Modal(this.$refs.modal)
    },
    methods : {
        close(){
            this.modal.hide()
            this.$emit('close');
        },
        show(){
            this.modal.show()
        }
    }
    // setup() {
        
    // },
}
</script>
<style scoped>
/* .modal-active{
	display:block;
} */
</style>