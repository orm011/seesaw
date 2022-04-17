<script setup>
import { onMounted } from 'vue'
import { counter_list } from './state'
import MTest from './m-test.vue'


onMounted(() => {
    console.log('counter_list', counter_list);
});

</script>

<template>
<MTest v-for="(s,idx) in counter_list.counts" :counter="s"></MTest>
<MTest v-for="(s,idx) in counter_list.counts" :counter="s"></MTest>
</template>