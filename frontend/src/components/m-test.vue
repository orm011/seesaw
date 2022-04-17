<script setup>
import { onMounted, defineProps } from 'vue'
import {state} from './state'

const props = defineProps(['counter']);

onMounted(() => {
  console.log(`the component is now mounted.`, props);
})

// props.counter.increment();

</script>

<template >
<span>State count: {{counter.count}} </span>
<button @click="counter.increment()">Incr</button>
<button @click="counter.count = 0">Reset</button>
</template>