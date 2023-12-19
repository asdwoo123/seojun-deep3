<template>
  <div class="overall-layout">
    <div class="image-frame">
        <img :src="predictStreamUrl" alt="">
    </div>
    <div>
        <div class="tool-layout">
            <p>Select model: </p>
            <a-dropdown>
                <template #overlay>
                    <a-menu @click="changeModel">
                        <a-menu-item v-for="(modelName, index) in modelNames" :key="index">
                            {{  modelName  }}
                        </a-menu-item>
                    </a-menu>
                </template>
                <a-button class="select-model-btn">
                    {{  selectModelName  }}
                    <DownOutlined />
                </a-button>    
            </a-dropdown>
        </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import axios from 'axios'

const serverUrl = 'http://192.168.0.48:5000'

const getModelNamesUrl = `${serverUrl}/model_list`
const changeModelUrl = `${serverUrl}/change_model`
const predictStreamUrl = ref([`${serverUrl}/predict`])
const selectModelName = ref('')


const modelNames = ref([])

async function getModelNames() {
    const response = await axios.get(getModelNamesUrl)
    modelNames.value = response.data
    selectModelName.value = modelNames.value[0]
}

function changeModel(e) {
    const key = e.key
    selectModelName.value = modelNames.value[key]
    axios.post(changeModelUrl, {
        modelName: selectModelName.value
    })
}

onMounted(() => {
    getModelNames()
})

</script>

<style>
.select-model-btn {
    margin-left: 30px;
}
</style>