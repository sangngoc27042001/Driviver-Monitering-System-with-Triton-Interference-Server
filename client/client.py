# Copyright 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from tritonclient.utils import *
import tritonclient.http as httpclient
import sys
import cv2
import numpy as np
import time 
from config import config
from Alert_by_counting_frames import Alert_by_counting_frames

alerter = Alert_by_counting_frames()

def call_API(input, model_name, inpu_type):

    with httpclient.InferenceServerClient("localhost:8000") as client:
        input0_data = input.astype(inpu_type)
        inputs = [
            httpclient.InferInput("INPUT0", input0_data.shape,
                                np_to_triton_dtype(input0_data.dtype)),
        ]

        inputs[0].set_data_from_numpy(input0_data)

        outputs = [
            httpclient.InferRequestedOutput("OUTPUT0"),
        ]

        response = client.infer(model_name,
                                inputs,
                                request_id=str(1),
                                outputs=outputs)

        result = response.get_response()
        output0_data = response.as_numpy("OUTPUT0")

        return output0_data



image = cv2.imread('./test.jpg')
image = cv2.flip(image, 1)

start = time.time()

arrayPoint = call_API(image, "get_multile_face_landmarks", np.uint8)
print(f'get_multile_face_landmarks: {time.time()-start}')
start = time.time()
print(arrayPoint.shape)

xyxy_phone_cigarette = call_API(image, "get_xyxy_phone_cigarette", np.uint8)
print(f'get_multile_face_landmarks: {time.time()-start}')
start = time.time()
print(xyxy_phone_cigarette.shape)



alerter.recieve_values_phone_cigarette(image, xyxy_phone_cigarette)

if arrayPoint.shape == (478, 3):

    ear_mar = call_API(arrayPoint, "return_variables", np.float32)
    print(f'return_variables: {time.time()-start}')
    start = time.time()
    print(ear_mar)
    ear, mar = ear_mar[0], ear_mar[1]

    alerter.recieve_values_ear_mar(image, ear, mar)



