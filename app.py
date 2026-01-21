import gradio as gr
from transformers import AutoProcessor, AutoModelForImageTextToText, TextIteratorStreamer
from transformers.image_utils import load_image
from threading import Thread
import torch
import pickle as pkl
import re 
from PIL import Image
import json
# import spaces  # Not needed for Colab
import os
import numpy as np
# from serve_constants import html_header, bibtext, learn_more_markdown, tos_markdown  # Commented out for Colab

# cur_dir = os.path.dirname(os.path.abspath(__file__))  # Not needed for Colab

# For running on Colab

MODEL_ID = "TIGER-Lab/PixelReasoner-RL-v1"

do_controlled_rectify = False  # Set to False for normal operation

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True,
                                          max_pixels=512*28*28)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
).to("cuda").eval()

def zoom(image, bbox_2d,padding=(0.1,0.1)):
    """
    Crop the image based on the bounding box coordinates.
    """
    img_x, img_y = image.size
    padding_tr = (600.0/img_x,600.0/img_y)
    padding = (min(padding[0],padding_tr[0]),min(padding[1],padding_tr[1]))

    if bbox_2d[0] < 1 and bbox_2d[1] < 1 and bbox_2d[2] < 1 and bbox_2d[3] < 1:
        normalized_bbox_2d = (float(bbox_2d[0])-padding[0], float(bbox_2d[1])-padding[1], float(bbox_2d[2])+padding[0], float(bbox_2d[3])+padding[1])
    else:
        normalized_bbox_2d = (float(bbox_2d[0])/img_x-padding[0], float(bbox_2d[1])/img_y-padding[1], float(bbox_2d[2])/img_x+padding[0], float(bbox_2d[3])/img_y+padding[1])
    normalized_x1, normalized_y1, normalized_x2, normalized_y2 = normalized_bbox_2d
    normalized_x1 =min(max(0, normalized_x1), 1)
    normalized_y1 =min(max(0, normalized_y1), 1)
    normalized_x2 =min(max(0, normalized_x2), 1)
    normalized_y2 =min(max(0, normalized_y2), 1)
    cropped_img = image.crop((int(normalized_x1*img_x), int(normalized_y1*img_y), int(normalized_x2*img_x), int(normalized_y2*img_y)))
    w, h = cropped_img.size
    assert w > 28 and h > 28, f"Cropped image is too small: {w}x{h}"


    return cropped_img  


def execute_tool(images, rawimages, args, toolname, is_video, function=None):
    if toolname=='select_frames':
        tgt = args['target_frames']
        if len(tgt)>8:
            message = f"You have selected {len(tgt)} frames in total. Think again which frames you need to check in details (no more than 8 frames)"
            # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
            ##### controlled modification
            if do_controlled_rectify and np.random.uniform()<0.75:
                if np.random.uniform()<0.25:
                    tgt = tgt[:len(tgt)//2]
                elif np.random.uniform()<0.25/0.75:
                    tgt = tgt[-len(tgt)//2:]
                elif np.random.uniform()<0.25/0.5:
                    tgt = tgt[::2]
                else:
                    tgt = np.random.choice(tgt, size=len(tgt)//2, replace=False)
                    tgt = sorted(tgt)
                selected_frames = function(images[0], tgt)
                message = tgt
            else: 
                selected_frames = []
            # selected_frames = function(images[0], [x-1 for x in tgt][::2]) # video is always in the first item
        elif max(tgt)>len(images[0]):
            message = f"There are {len(images[0])} frames numbered in range [1,{len(images[0])}]. Your selection is out of range."
            selected_frames = []
        else:
            message = ""
            candidates = images[0]
            if not isinstance(candidates, list):
                candidates = [candidates]
            selected_frames = function(candidates, [x-1 for x in tgt]) # video is always in the first item
        return selected_frames, message
    else:
        tgt = args['target_image']
        if is_video:
            if len(images)==1: # there is only 
                # we default the candidate images into video frames 
                video_frames = images[0]
                index = tgt - 1 
                assert index<len(video_frames), f"Incorrect `target_image`. You can only select frames in the given video within [1,{len(video_frames)}]"
                image_to_crop = video_frames[index]
            else: # there are zoomed images after the video; images = [[video], img, img, img]
                cand_images = images[1:]
                index = tgt -1
                assert index<len(cand_images), f"Incorrect `target_image`. You can only select a previous frame within [1,{len(cand_images)}]"
                image_to_crop = cand_images[index]
        else:
            index =  tgt-1 
            assert index<len(images), f"Incorrect `target_image`. You can only select previous images within [1,{len(images)}]"
            
            if index<len(rawimages):
                tmp = rawimages[index]
            else:
                tmp = images[index]
            image_to_crop = tmp
        if function is None: function = zoom
        cropped_image = function(image_to_crop, args['bbox_2d'])
    return cropped_image
    

def parse_last_tool(output_text):
    # print([output_text])
    return json.loads(output_text.split(tool_start)[-1].split(tool_end)[0])

tool_end = '</tool_call>'
tool_start = '<tool_call>'

def model_inference(input_dict, history):
    text = input_dict["text"]
    files = input_dict["files"]

    """
    Create chat history
    Example history value:
    [
        [('pixel.png',), None], 
        ['ignore this image. just say "hi" and nothing else', 'Hi!'], 
        ['just say "hi" and nothing else', 'Hi!']
    ]
    """
    all_images = []
    current_message_images = []
    sysprompt = "<|im_start|>system\nYou are a helpful assistant.\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"crop_image_normalized\", \"description\": \"Zoom in on the image based on the bounding box coordinates.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"bbox_2d\": {\"type\": \"array\", \"description\": \"normalized coordinates for bounding box of the area you want to zoom in. Values within [0.0,1.0].\", \"items\": {\"type\": \"number\"}}, \"target_image\": {\"type\": \"number\", \"description\": \"The index of the image to crop. Index from 1 to the number of images. Choose 1 to operate on original image.\"}}, \"required\": [\"bbox_2d\", \"target_image\"]}}}\n{\"type\": \"function\", \"function\": {\"name\": \"select_frames\", \"description\": \"Select frames from a video.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"target_frames\": {\"type\": \"array\", \"description\": \"List of frame indices to select from the video (no more than 8 frames in total).\", \"items\": {\"type\": \"integer\", \"description\": \"Frame index from 1 to 16.\"}}}, \"required\": [\"target_frames\"]}}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>"
    messages = [{
        "role": "user",
        "content": sysprompt
    }]
    hint = "\n\nGuidelines: Understand the given visual information and the user query. Determine if it is beneficial to employ the given visual operations (tools). For a video, we can look closer by `select_frames`. For an image, we can look closer by `crop_image_normalized`. Reason with the visual information step by step, and put your final answer within \\boxed{}."
    for val in history:
        if val[0]:
            if isinstance(val[0], str):
                messages.append({
                    "role": "user", 
                    "content": [
                        *[{"type": "image", "image": image} for image in current_message_images],
                        {"type": "text", "text": val[0]},
                    ],
                })
                current_message_images = []

            else:
                # Load messages. These will be appended to the first user text message that comes after
                current_message_images = [load_image(image) for image in val[0]]
                all_images += current_message_images

        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})
    
    imagelist = rawimagelist = current_message_images = [load_image(image) for image in files]
    all_images += current_message_images
    messages.append({
        "role": "user",
        "content": [
            *[{"type": "image", "image": image} for image in current_message_images],
            {"type": "text", "text": text+hint},
        ],
    })
    
    print(messages)

    complete_assistant_response_for_gradio = []
    while True:
        """
        Generate and stream text
        """
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[prompt],
            images=all_images if all_images else None,
            return_tensors="pt",
            padding=True,
        ).to("cuda")
        print(f"===> messages for generation")
        print(messages)
        streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=False)
        generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=1024, temperature=0.1, top_p=0.95, top_k=50)
        # generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=1024, do_sample=False, num_beams=1)
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        current_model_output_segment = "" # Text generated in this specific model call
        toolflag = False
        for new_text_chunk in streamer:
            current_model_output_segment += new_text_chunk
            # Yield the sum of previously committed full response parts + current streaming segment
            # yield complete_assistant_response_for_gradio + current_model_output_segment
            if tool_start in current_model_output_segment:
                toolflag = True
                tmp = current_model_output_segment.split(tool_start)[0]
                yield complete_assistant_response_for_gradio + [tmp+"\n\n<b>Planning Visual Operations ...</b>\n\n"]
            if not toolflag:
                yield complete_assistant_response_for_gradio + [current_model_output_segment]
        thread.join()

        # Process the full segment (e.g., remove <|im_end|>)
        processed_segment = current_model_output_segment.split("<|im_end|>", 1)[0] if "<|im_end|>" in current_model_output_segment else current_model_output_segment
        # messages.append(dict(role='assistant', content=processed_segment))
        # Append this processed segment to the cumulative display string for Gradio
        complete_assistant_response_for_gradio += [processed_segment + "\n\n"]
        yield complete_assistant_response_for_gradio # Ensure the fully processed segment is yielded to Gradio

        
        # Check for tool call in the *just generated* segment
        qatext_for_tool_check = processed_segment
        require_tool = tool_end in qatext_for_tool_check and tool_start in qatext_for_tool_check
                
        if require_tool:
            
            tool_params = parse_last_tool(qatext_for_tool_check) 
            tool_name = tool_params['name']
            tool_args = tool_params['arguments']
            # complete_assistant_response_for_gradio += f"\n<b>Executing Visual Operations ...</b> @{tool_name}({tool_args})\n\n"
            complete_assistant_response_for_gradio += [f"\n<b>Executing Visual Operations ...</b> @{tool_name}({tool_args})\n\n"]
            yield complete_assistant_response_for_gradio # Update Gradio display
            video_flag = False
            print(f"candidate images", all_images)
            raw_result = execute_tool(all_images, all_images, tool_args, tool_name, is_video=video_flag)
            print(raw_result)
            proc_img = raw_result
            all_images += [proc_img]
            proc_img.save("tmp.png")
            display = [dict(text="", files=["tmp.png"])]
            complete_assistant_response_for_gradio = complete_assistant_response_for_gradio + display
            yield complete_assistant_response_for_gradio # Update Gradio display
            
            new_piece = dict(role='user', content=[
                                    dict(type='text', text="\nHere is the cropped image (Image Size: {}x{}):".format(proc_img.size[0], proc_img.size[1])),
                                    dict(type='image', image=proc_img)
                                ]
            )
            messages.append(new_piece)
            # complete_assistant_response_for_gradio += f"\n<b>Analyzing Operation Result ...</b> @region(size={proc_img.size[0]}x{proc_img.size[1]})\n\n"
            complete_assistant_response_for_gradio += [f"\n<b>Analyzing Operation Result ...</b> @region(size={proc_img.size[0]}x{proc_img.size[1]})\n\n"]
            yield complete_assistant_response_for_gradio # Update Gradio display

            
        else:
            break

with gr.Blocks() as demo:
    examples = [
        [
            {
                "text": "What kind of restaurant is it?", 
                "files": [
                    "1.jpg"
                ]
            }
        ]
    ]

    # gr.HTML(html_header)  # Commented out for Colab
    
    # image_op_display = gr.Image(label="Visual Operation Result", type="pil", height=480, show_download_button=True, interactive=False)

    gr.ChatInterface(
        fn=model_inference,
        description="# **Pixel Reasoner**",
        chatbot=gr.Chatbot(label="Conversation", layout="bubble", bubble_full_width=False, show_copy_button=True, height=600),
        examples=examples,
        # fill_height=True,
        textbox=gr.MultimodalTextbox(label="Query Input", file_types=["image"], file_count="multiple"),
        stop_btn="Stop Generation",
        multimodal=True,
        cache_examples=False,
    )
    
    # gr.Markdown(tos_markdown)  # Commented out for Colab
    # gr.Markdown(learn_more_markdown)  # Commented out for Colab
    # gr.Markdown(bibtext)  # Commented out for Colab

demo.launch(share=True)  # share=True works well with Colab
