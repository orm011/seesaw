<template>
  <div>
    <p>vue-json-editor</p>
    <Vue3JsonEditor
      v-model="json"
      :show-btns="true"
      :expandedOnStart="true"
      @json-change="onJsonChange"
    />
  </div>
</template>

<script>
import { defineComponent, reactive, toRefs } from 'vue'
import { Vue3JsonEditor } from 'vue3-json-editor'



export default defineComponent({
  components: {
    Vue3JsonEditor, 
  },
  props: ['default_params'], 
  setup (props) {
    function onJsonChange (value) {
      console.log('value:', value)
      state.json = value; 
    }
    function currentConfig(){
      return state.json; 
    }
    function updateClientData(default_params){
      console.log("Update called in config"); 
      state.json = default_params; 
    }
 
    const state = reactive({
      json: props.default_params,
    });

    return {
      ...toRefs(state),
      onJsonChange, 
      currentConfig,
      updateClientData, 
    }
  }
})
</script>