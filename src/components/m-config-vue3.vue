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
  props: ['client_data'], 
  setup (props) {
    function onJsonChange (value) {
      console.log('value:', value)
      state.json = value; 
    }
    function currentConfig(){
      return state.json; 
    }
    function updateClientData(client_data){
      console.log("Update called in config"); 
      if (client_data.session === null){
        console.log("default params"); 
        state.json = client_data.default_params; 
      } else { 
        console.log("current params"); 
        state.json = client_data.session.params; 
      }
    }
 
    var state; 
    if (props.client_data.session === null){
      console.log("default params"); 
      state = reactive({
        json: props.client_data.default_params,
      });
    } else { 
      console.log("current params"); 
      state = reactive({
        json: props.client_data.session.params, 
      });
    }

    return {
      ...toRefs(state),
      onJsonChange, 
      currentConfig,
      updateClientData, 
    }
  }
})
</script>