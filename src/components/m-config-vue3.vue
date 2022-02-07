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
  props: [session], 
  setup () {
    function onJsonChange (value) {
      console.log('value:', value)
    }

    if (session === null){
      const state = reactive({
        json: {'batch_size': 64,
                    'logit_scale_init': 3.7,
                    'image_loss_weight':1.,
                    'device':'cuda:0',
                    'opt_config': {'logit_scale': null, 
                      'transformer': null,
                      'transformer.resblocks.0.ln_': {'lr': 0.0007435612322566577,'weight_decay': 1.5959136512232553e-05},
                      'transformer.resblocks.11.ln': {'lr': 0.0001298217305130271,'weight_decay': 0.015548602355938877},
                      'ln_final': {'lr': 0.007707377565843718,'weight_decay': 0.0},
                      'text_projection': {'lr': 5.581683501371101e-05, 'weight_decay': 0.0},
                      'positional_embedding':null,
                      'token_embedding':null,
                      'visual': null,
                      'positiional_embedding':null},
                    'num_warmup_steps': 5,
                    'rounds': 4,
                    'margin':.3,
                    'num_workers': 20,
                    'test_size': 1000,
                    'val_batch_size': 500}
      }); 
    } else { 
      const state = reactive({
        json: {'batch_size': 64,
                    'logit_scale_init': 3.7,
                    'image_loss_weight':1.,
                    'device':'cuda:0',
                    'opt_config': {'logit_scale': null, 
                      'transformer': null,
                      'transformer.resblocks.0.ln_': {'lr': 0.0007435612322566577,'weight_decay': 1.5959136512232553e-05},
                      'transformer.resblocks.11.ln': {'lr': 0.0001298217305130271,'weight_decay': 0.015548602355938877},
                      'ln_final': {'lr': 0.007707377565843718,'weight_decay': 0.0},
                      'text_projection': {'lr': 5.581683501371101e-05, 'weight_decay': 0.0},
                      'positional_embedding':null,
                      'token_embedding':null,
                      'visual': null,
                      'positiional_embedding':null},
                    'num_warmup_steps': 5,
                    'rounds': 4,
                    'margin':.3,
                    'num_workers': 20,
                    'test_size': 1000,
                    'val_batch_size': 500}
      });
    }

    return {
      ...toRefs(state),
      onJsonChange
    }
  }
})
</script>