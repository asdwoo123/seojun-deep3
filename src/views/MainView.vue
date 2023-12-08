<template lang="ko">
    <div>
        <div style="display: flex;">
            <div class="image-frame" @mousedown="handleMouseDown" @mousemove="handleMouseMove" @mouseup="handleMouseUp">
                <img :src="(predict) ? 'http://localhost:5000/predict' : thumbnails[selectIndex].imagePath" alt="stream" @click="selectCancel">
                <vue-draggable-resizable style="background-color: rgba(255, 0, 0, 0.3);" v-for="({x, y, w, h}, index) in thumbnails[selectIndex].selectionBoxes" class="select" :w="w" :h="h" :x="x" :y="y" @dragging="onDrag" @resizing="onResize" @activated="selectSelection(index)"
                    :parent="true" />
                <div class="selection" v-if="isSelection && endX !== 0" :style="{
                                top: ((startY > endY) ? endY : startY) + 'px',
                                left: ((startX > endX) ? endX : startX) + 'px',
                                width: Math.abs(endX - startX) + 'px',
                                 height: Math.abs(endY - startY) + 'px'
                             }" />
            </div>
            <div class="tool-layout">
                <div>
                    <button class="tool-button" @click="mode = 'selection'" type="button">selection mode</button>
                    <button class="tool-button" @click="mode = 'edit'" type="button">edit mode</button>
                </div>
                <div>
                    <button class="tool-button" @click="deleteThum" type="button">delete</button>
                    <button class="tool-button" @click="saveThum" type="button">save</button>
                </div>
                <div>
                    <button @click="train" class="tool-button" type="button">train</button>
                    <button class="tool-button"
                    @click="predict = !predict"
                     type="button">{{ (predict) ? 'create' : 'prediction' }}</button>
                </div>
                <div class="model-path-layout" style="height: 21px;">
                    label no:
                    <input type="number" v-if="isSelect" v-model="thumbnails[selectIndex].selectionBoxes[labelIndex].no">
                </div>
                <div class="model-path-layout">
                    model folder path: {{ saveModelPath }}
                    <button type="button">select</button>
                </div>
                <div class="model-path-layout">
                    load model
                </div>
                <p style="text-align: center; font-size: 20px;">
                    {{ mode + ' mode' }}
                </p>
            </div>
        </div>
        <div class="thumbnail-layout">
            <div v-for="(thumbnail, index) in thumbnails" class="thumbnail" @click="() => {selectIndex = index; selectCancel();}" :style="
                        (selectIndex == index) ? {
                            border: '3px solid green'
                        } : ''">
                <img width="150" height="150" :src="thumbnail.imagePath" alt="">
                <p>{{ thumbnail.imageName }}</p>
            </div>
        </div>
    </div>
</template>
<script>
import VueDraggableResizable from 'vue-draggable-resizable'
import 'vue-draggable-resizable/dist/VueDraggableResizable.css'
import axios from 'axios'
import { io } from 'socket.io-client'
import path from 'path'
import fs from 'fs'
export default {
    data() {
        return {
            db: null, 
            message: '',
            labelIndex: null,
            selectIndex: 0,
            prevKey: null,
            mode: 'edit',
            isPusing: false,
            isSelection: false,
            isSelect: false,
            socket: null,
            isCopying: false,
            assetsPath: '',
            saveModelPath: '',
            loadModelPath: '',
            predict: false,
            cameraUrl: 'http://192.168.0.178:3000/camera',
            startX: 0,
            startY: 0,
            endX: 0,
            endY: 0,
            datasets: [{
                imageName: 'stream',
                imagePath: 'http://192.168.0.178:3000/camera/stream',
                selectionBoxes: []
            }],
            thumbnails: [{
                imageName: 'stream',
                imagePath: 'http://192.168.0.178:3000/camera/stream',
                selectionBoxes: []
            }]
        }
    },
    components: {
        VueDraggableResizable
    },
    mounted() { },
    created() {
        window.addEventListener('keydown', this.handleKeyDown)
        window.addEventListener('keyup', this.handleKeyUp)
        const assetsPath = path.resolve('src/assets')
        this.assetsPath = assetsPath
        const socket = io('http://192.168.0.48:5000')
        this.socket = socket
        socket.on('connect', (socket) => {
            console.log('socket connect!')
        });

        socket.on('train', (message) => {
            console.log(message);
        });
    },
    methods: {
        train() {
            let data = this.thumbnails.slice(1)
            data = data.map((thum) => {
                const image = thum.buffer
                const labels = thum.selectionBoxes.map((bs) => {
                    return [bs.no, bs.x, bs.y, bs.x + bs.w, bs.y + bs.h]
                })
                return [image, labels]
            });
            console.log(data)
            if (!this.socket) return
            this.socket.emit('train', {
                data
            })
        },
        handleMouseDown(event) {
            if (this.mode == 'edit') return;
            this.isSelection = true;
            this.endX = 0;
            this.startX = event.offsetX;
            this.startY = event.offsetY;
        },
        handleMouseMove(event) {
            if (event.target.tagName !== 'IMG') return;
            this.endX = event.offsetX;
            this.endY = event.offsetY;
        },
        handleMouseUp() {
            this.isSelection = false;
            if (this.mode == 'edit') return;
            this.thumbnails[this.selectIndex].selectionBoxes = [...this.thumbnails[this.selectIndex].selectionBoxes, {
                y: ((this.startY > this.endY) ? this.endY : this.startY),
                x: ((this.startX > this.endX) ? this.endX : this.startX),
                w: Math.abs(this.endX - this.startX),
                h: Math.abs(this.endY - this.startY),
                no: 0
            }]
            this.mode = 'edit'
        },
        handleKeyDown(event) {
            const key = event.key
            if (key === 'Delete') this.deleteLabel()
            if (key === 'Alt') this.isCopying = true
        },
        handleKeyUp(event) {
            const key = event.key
            if (key === 'Alt') this.isCopying = false
        },
        handleKeyPress(event) {
            const key = event.key
        },
        selectSelection(index) {
            this.isSelect = true
            this.labelIndex = index
            if (this.isCopying) {
                this.thumbnails[this.selectIndex].selectionBoxes = [...this.thumbnails[this.selectIndex].selectionBoxes, {
                    ...this.thumbnails[this.selectIndex].selectionBoxes[this.labelIndex]
                }]
            }
        },
        selectCancel() {
            console.log('cancel')
            this.isSelect = false
        },
        deleteLabel() {
            if (this.labelIndex === null || !this.isSelect) return
            this.thumbnails[this.selectIndex].selectionBoxes.splice(this.labelIndex, 1)
        },
        copyLabel() {
            if (this.labelIndex === null || !this.isSelect) return
        },
        deleteThum() {
            this.thumbnails.splice(this.selectIndex, 1)
        },
        saveThum() {
            axios.get(`${this.cameraUrl}/capture`, {
                responseType: 'arraybuffer'
            }).then((response) => {
                const buffer = Buffer.from(response.data, 'binary');
                if (this.selectIndex === 0) {
                    const imageName = `labeling${this.thumbnails.length}`
                    fs.writeFileSync(`${this.assetsPath}/images/${imageName}.jpg`, buffer)
                    const base64Image = buffer.toString('base64')
                    const selectionBoxes = [...this.thumbnails[this.selectIndex].selectionBoxes]
                    this.thumbnails.push({
                        imageName,
                        buffer,
                        imagePath: `data:image/jpeg;base64, ${base64Image}`,
                        selectionBoxes
                    })
                    this.thumbnails[0].selectionBoxes = []
                }
            });
        },
        onResize(x, y, w, h) {
            this.thumbnails[this.selectIndex].selectionBoxes[this.labelIndex] = {
                ...this.thumbnails[this.selectIndex].selectionBoxes[this.labelIndex],
                x,
                y,
                w,
                h
            }
        },
        onDrag(x, y) {
            this.thumbnails[this.selectIndex].selectionBoxes[this.labelIndex] = {
                ...this.thumbnails[this.selectIndex].selectionBoxes[this.labelIndex],
                x,
                y,
            }
        }
    }
}
</script>
<style scoped>
.image-frame {
    position: relative;
    border: 1px solid red;
    width: 640px;
    height: 640px;
}

.image-frame>img {
    position: absolute;
    top: 0;
    left: 0;
    -webkit-user-drag: none;
}

.selection {
    position: absolute;
    border: 1px solid greenyellow;
}

.tool-layout {
    display: flex;
    flex-direction: column;
}

.tool-layout>div {
    display: flex;
}

.tool-button {
    margin: 20px;
    width: 200px;
    height: 60px;
}

.model-path-layout {
    margin: 10px 0 0 20px;
    justify-content: space-between;
}

.thumbnail-layout {
    display: flex;
}

.thumbnail {
    margin: 10px;
}</style>
