<template>
  <div class="overall-layout">
    <div>
        <div class="image-frame" @mousedown="handleMouseDown" @mousemove="handleMouseMove" @mouseup="handleMouseUp">
            <img :src="selectDataset.imagePath" alt="">
            <div class="create-box" v-if="isClicking && endX !== 0"
            :style="draggableInfo"
             />
             <vue-draggable-resizable
              v-for="({x, y, w, h}, index) in selectBboxes"
              :style="bboxStyle(selectBboxes[index].no)"
             :key="index" :w="w" :h="h" :x="x" :y="y" :parent="true"
             @activated="activateBbox(index)" @deactivated="deactiveBbox" @resizing="onResize" @dragging="onDrag"
              />
        </div>
        <div class="preview-layout">
            <div class="preview" :key="index" v-for="(dataset, index) in datasets"
             @click="activateDataset(index)">
                <img :src="dataset.imagePath" :style="previewActivateStyle(index)" alt="preview" />
                <p>{{ dataset.name }}</p>
            </div>
        </div>
    </div>

    <div>
        <div>
            <a-button danger class="tool-button" @click="deleteDataset">Delete dataset</a-button>
            <a-button danger class="tool-button" @click="createDataset">Create dataset</a-button>
        </div>
        <div class="tool-layout" v-if="selectBboxes.length > 0">
            <p>no: </p>
            <a-input-number id="inputNumber" v-model:value="selectBbox.no" :min="0" :max="10" />
        </div>
        <div class="tool-layout">
          <p>save model name: </p>  
          <a-input v-model:value="modelName" placeholder="Save model name" />
        </div>
        <div style="height: 20px;" />
        <div class="tool-layout">
            <div class="train-option">
                <p style="margin-bottom: 10px;">Epoch</p>
                <a-input-number v-model:value="trainEpoch" :min="1" :max="100" />
            </div>
            <div class="train-option">
                <p style="margin-bottom: 10px;">Batch size</p>
                <a-input-number v-model:value="batchSize" :min="1" :max="10" />
            </div>
        </div>
        <div class="train-layout">
            <a-button type="primary" @click="train">
                Train
            </a-button>
        </div>
    </div>
    <a-modal v-model:open="isTraining" 
    :footer="null" :centered="true">
        <div style="padding: 50px 0">
            <a-progress :percent="percentage" :size="[300, 20]" />
            <div style="height: 20;" />
            <p>{{ trainMessage }}</p>
        </div>
    </a-modal>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import axios from 'axios'
import { io } from 'socket.io-client'


const cameraUrl = 'http://192.168.0.178:3000/camera'
const streamUrl = `${cameraUrl}/stream`
const captureUrl = `${cameraUrl}/capture`
const socketUrl = 'http://192.168.0.48:5000'
const sampleUrl = 'https://upload.wikimedia.org/wikipedia/commons/e/ee/Sample_abc.jpg'

const datasets = ref([{
    name: 'stream',
    imageBuffer: [],
    imagePath: sampleUrl,
    bbox: []
}])
const isClicking = ref(false)
const startX = ref(0)
const startY = ref(0)
const endX = ref(0)
const endY = ref(0)
const datasetIndex = ref(0)
const bboxIndex = ref(0)
const modelName = ref('')
const socket = ref({})
const isTraining = ref(false)
const trainMessage = ref('')
const trainStep = ref(0)
const percentage = ref(0)
const trainEpoch = ref(50)
const batchSize = ref(2)

const draggableInfo = computed(() => ({
    top: ((startY.value > endY.value) ? endY.value : startY.value) + 'px',
    left: ((startX.value > endX.value) ? endX.value : startX.value) + 'px',
    width: Math.abs(endX.value - startX.value) + 'px',
    height: Math.abs(endY.value - startY.value) + 'px'
}))

const selectDataset = computed(() => datasets.value[datasetIndex.value])
const selectBboxes = computed(() => selectDataset.value.bbox)
const selectBbox = computed({
    get: () => selectBboxes.value[bboxIndex.value],
    set: (val) => {
        let bbox = datasets.value[datasetIndex.value].bbox[bboxIndex.value]
        bbox = {
            ...bbox,
            ...val
        }
        console.log(bbox)
    }
})



function handleKeyDown(event) {
    const key = event.key
    if (key === 'Delete') deleteBbox()
}

function bboxStyle(no) {
    return {
        backgroundColor: `red`,
        opacity: 0.3
    }
}

function activateBbox(index) {
    bboxIndex.value = index
}

function activateDataset(index) {
    datasetIndex.value = index
}

function deactiveBbox() {
    console.log('deactive')
}

function previewActivateStyle(index) {
    return (datasetIndex.value == index) ? {
        border: '3px solid red'
    } : ''
}

function checkCreateMode(event) {
    return event.target.tagName === 'IMG'
}

function handleMouseDown(event) {
    if (!checkCreateMode(event)) return
    isClicking.value = true
    startX.value = event.offsetX
    startY.value = event.offsetY
}

function handleMouseMove(event) {
    if (!checkCreateMode(event) || !isClicking.value) return
    endX.value = event.offsetX
    endY.value = event.offsetY
}

function handleMouseUp(event) {
    if (!checkCreateMode(event) || !isClicking.value) return
    const x = ((startX.value > endX.value) ? endX.value : startX.value)
    const y = ((startY.value > endY.value) ? endY.value : startY.value)
    const w = Math.abs(endX.value - startX.value)
    const h = Math.abs(endY.value - startY.value)
    if (w > 30 && h > 30) {
        selectBboxes.value.push({
        no: 0,
        x,
        y,
        w,
        h, 
    })
    }
    isClicking.value = false
    startX.value = 0
    startY.value = 0
    endX.value = 0
    endY.value = 0
}

function onResize(x, y, w, h) {
    selectBbox.value = {
        x,
        y,
        w,
        h
    }
}

function onDrag(x, y) {
    selectBbox.value = {
        x,
        y
    }

}

async function getCaptureImage(url) {
    const response = await axios.get(url, {
        responseType: 'arraybuffer'
    })

    const buffer = Buffer.from(response.data, 'binary')
    return buffer
}

async function createDataset() {
    if (datasetIndex.value !== 0) return
    const buffer = await getCaptureImage(sampleUrl)
    const base64Image = buffer.toString('base64')
    const base64Path = `data:image/jpeg;base64, ${base64Image}`
    const name = `data${datasets.value.length}`
    datasets.value.push({
        name,
        imageBuffer: buffer,
        imagePath: base64Path,
        bbox: [...selectBboxes.value]
    })
    
    datasets.value[0].bbox = []
}

function deleteBbox() {
    if (selectBboxes.value.length < 1) return
    selectBboxes.value.splice(bboxIndex.value, 1)
}

function deleteDataset() {
    if (datasetIndex.value === 0) return
    datasets.value.splice(datasetIndex, 1)
}

function train() {
    let datasets = datasets.value.slice(1)
    datasets = datasets.map((dataset) => {
        const image = dataset.imageBuffer
        const bboxes = dataset.bbox.map((bs) => 
        [bs.no, bs.x, bs.y, bs.x + bs.w, bs.y + bs.h])
        return [image, bboxes]
    })

    if (socket.value.connected) {
        isTraining.value = true
        socket.value.emit('train', { data: datasets })
    }
}

onMounted(() => {
    window.addEventListener('keydown', handleKeyDown)
    socket.value = io(socketUrl)

    console.log(socket.value)
    
    socket.value.on('connect', () => {
        console.log('socketio connect!');
    })

    socket.value.on('message', (msg) => {
        trainMessage.value = msg
    })

    socket.value.on('step', (step) => {
        if (cnt > 99) isTraining.value = false
        step.value = step
    })

    socket.value.on('count', (cnt) => {
        if (cnt > 99) isTraining.value = false
        percentage.value = cnt 
    })
})

</script>

<style lang="scss">
.image-frame {
    position: relative;
    width: 640px;
    height: 640px;

    img {
        width: 100%;
        height: 100%;
        position: absolute;
        top: 0;
        left: 0;
        -webkit-user-drag: none;
    }
}

.tool-button {
    margin: 0 0 20px 20px;
    width: 200px;
    height: 60px;
}

.bbox {
    background-color: rgba(255, 0, 0, 0.3);
}

.create-box {
    position: absolute;
    border: 1px solid greenyellow;
}

.draggable-resizble-box {
    background-color: rgba(255, 0, 0, 0.3);
}

.preview-layout {
    display: flex;
}

.preview {
    margin: 10px 10px 10px 0;

    p {
        margin-top: 10px;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    
    img {
        width: 150px;
        height: 150px;
    }
}

.train-layout {
    display: flex;
    justify-content: flex-end;
    margin-top: 30px;

    button {
        width: 200px;
        height: 70px;
    }
}

</style>