<template>
  <div class="image-frame">
    train
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import axios from 'axios'
import path from 'path'
import fs from 'fs'

const datasets = [{
    name: 'stream',
    imagePath: 'http://via.placeholder.com/640x640',
    bbox: []
}]

const isEditMode = ref(false)
const isClicking = ref(false)
const startX = ref(0)
const startY = ref(0)
const endX = ref(0)
const endY = ref(0)

const draggableInfo = computed(() => ({
    disable: isClicking.value && endX.value !== 0,
    top: ((startY.value > endY.value) ? endY.value : startY.value) + 'px',
    left: ((startX.value > endX.value) ? endX.value : startX.value) + 'px',
    width: Math.abs(endX.value - startX.value) + 'px',
    height: Math.abs(endY.value - startY.value) + 'px'
}))

function handleKeyDown() {

}

function handleKeyUp() {

}

function handleMouseMove(event) {

}

function handleMouseUp() {

}

onMounted(() => {

})

</script>

<style>

</style>