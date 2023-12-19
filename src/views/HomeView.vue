<template>
    <div class="home">
      <div>
        <a-radio-group v-model:value="mode">
          <a-radio-button @click="onTapClick('train')" value="train">Train</a-radio-button>
          <a-radio-button @click="onTapClick('predict')" value="predict">Predict</a-radio-button>
        </a-radio-group>
        <div style="height: 15px;" />
      </div>
        <router-view/>
    </div>
</template>

<script setup>
import { ref } from 'vue'
import { useRouter } from 'vue-router'

const router = useRouter()
const mode = ref('train')

function onTapClick(value) {
  router.push(`/${value}`)
}

</script>
