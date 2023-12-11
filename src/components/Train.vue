<template>
  <div style="display: flex;">
    <div>
        <div class="image-frame" @mousedown="handleMouseDown" @mousemove="handleMouseMove" @mouseup="handleMouseUp">
            <img :src="selectDataset.imagePath" alt="">
            <div class="create-box" v-if="isCreateMode && endX !== 0"
            :style="draggableInfo"
             />
        </div>
    </div>

    <div>
        <div>
            <a-button danger class="tool-button" @click="changeCreateMode(true)">Create mode</a-button>
            <a-button danger class="tool-button" @click="changeCreateMode(false)">Edit mode</a-button>
        </div>
        <div>
            <a-button danger class="tool-button">Delete</a-button>
            <a-button danger class="tool-button">Save</a-button>
        </div>
    </div>
    
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import axios from 'axios'
import path from 'path'
import fs from 'fs'
import { start } from 'repl';
import { eventNames } from 'process';


const datasets = ref([{
    name: 'stream',
    imagePath: 'http://via.placeholder.com/640x640',
    bbox: []
}])
const isCreateMode = ref(false)
const isClicking = ref(false)
const startX = ref(0)
const startY = ref(0)
const endX = ref(0)
const endY = ref(0)
const datasetIndex = ref(0)
const bboxIndex = ref(0)

const draggableInfo = computed(() => ({
    // disable: isClicking.value && endX.value !== 0,
    top: ((startY.value > endY.value) ? endY.value : startY.value) + 'px',
    left: ((startX.value > endX.value) ? endX.value : startX.value) + 'px',
    width: Math.abs(endX.value - startX.value) + 'px',
    height: Math.abs(endY.value - startY.value) + 'px'
}))

const selectDataset = computed(() => datasets.value[datasetIndex.value])
const selectBbox = computed({
    get() {
        return selectDataset.value.bbox[bboxIndex.value]
    }
})

function handleKeyDown() {

}

function handleKeyUp() {

}

function changeCreateMode(value) {
    console.log(isCreateMode.value);
    isCreateMode.value = value
}

function handleMouseDown(event) {
    if (!isCreateMode.value) return
    isClicking.value = true
    startX.value = event.offsetX
    startY.value = event.offsetY
}

function handleMouseMove(event) {
    if (event.target.tagName !== 'IMG' || !isClicking.value) return
    endX.value = event.offsetX
    endY.value = event.offsetY
}

function handleMouseUp() {
    isClicking.value= false
    if (!isCreateMode.value) return
    selectDataset.value.bbox = [
        ...selectDataset.value.bbox, {
            x: ((startX.value > endX.value) ? endX.value : startX.value),
            y: ((startY.value > endY.value) ? endY.value : startY.value),
            w: Math.abs(endX.value- startX.value),
            h: Math.abs(endY.value - startY.value), 
        }
    ]
    endX.value = 0
    endY.value = 0
}

function onResize(x, y, w, h) {
    
}

function onDrag(x, y) {

}

onMounted(() => {
    window.addEventListener('keydown', handleKeyDown)
    window.addEventListener('keyup', handleKeyUp)
})

</script>

<style>
.tool-button {
    margin: 20px;
    width: 200px;
    height: 60px;
}

.create-box {
    position: absolute;
    border: 1px solid greenyellow;
}
</style>