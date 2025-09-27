elif mode == "2":
  pos_num = tokenCounter(civitai_params["positive_prompt"])
  neg_num = tokenCounter(civitai_params["negative_prompt"])
  print("Token数")
  print(" - Positive : "+str(pos_num))
  print(" - Negative : "+str(neg_num))
  mult = 1+max(pos_num, neg_num) // 77
  print(" - multiple : "+str(mult))
  import os
  import torch
  import re
  from datetime import datetime, timedelta
  jst = datetime.utcnow() + timedelta(hours=9)
  timeDir = {
      "y":jst.strftime("%Y"),
      "m":jst.strftime("%m"),
      "d":jst.strftime("%d"),
      "t":jst.strftime("%H-%M-%S")
      }
  timestamp = jst.strftime("%Y%m%d_%H%M%S")
  if civitai_params["submode"] == "2":
    from PIL import Image, ImageOps
    from IPython.display import display


  base_dir = "/content/models"
  os.makedirs(base_dir, exist_ok=True)
  download_path = "/content/models/downloaded_model.safetensors"

  #※Driveのinfo.txtに書くための変数!!
  # modelName = "unknown"
  # if "Model" in locals():
  #   regex = r'([0-9]+)'
  #   regex2 = r'#(\d+)'
  #   if bool(re.match(regex, Model)):
  #     modelName = model_list_C[int(Model)-1][1]
  #   else:
  #     match = re.fullmatch(regex2, Model)
  #     if match:
  #       modelName = "model_id:" + match.group(1)
  #     else:
  #       modelName = Model


  if civitai_params["status"] == "1":
    regex = r'([0-9]+)'
    regex2 = r'#(\d+)'
    if bool(re.match(regex, civitai_params['url'])):
      civitai_params['url'] = model_list_C[int(civitai_params['url'])-1][0]
    else:
      match = re.fullmatch(regex2, civitai_params['url'])
      if match:
        civitai_params['url'] = 'https://civitai.com/api/download/models/'+match.group(1)

    model_url = civitai_params["url"]

    print(model_url)

    print(f"Downloading Checkpoint model to {download_path}...")
    !wget -O {download_path} "{model_url}"
    print(f"Download of safetensorFile complete!")


  lora_path = civitai_params["lora_path"] # LoRAはダウンロードしないとNone


  if civitai_params["status"] == "1" or civitai_params["status"] == "2":
    # Determine the checkpoint path for conversion
    conversion_checkpoint_path = civitai_params.get("model_insert_path")
    if not conversion_checkpoint_path or not os.path.exists(conversion_checkpoint_path):
        # If no insert path or path doesn't exist, use the downloaded path
        conversion_checkpoint_path = download_path

    if civitai_params["need_yaml"]:
      !wget -O /content/models/v1-inference.yaml \
https://raw.githubusercontent.com/CompVis/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml
      #!wget -q https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-inference.yaml
    if not civitai_params["sdxl"]:
      !mkdir /content/converted
      !wget -O convert_original_stable_diffusion_to_diffusers.py \
      https://raw.githubusercontent.com/huggingface/diffusers/main/scripts/convert_original_stable_diffusion_to_diffusers.py
      os.makedirs(base_dir, exist_ok=True)
      !python convert_original_stable_diffusion_to_diffusers.py \
        --checkpoint_path "{conversion_checkpoint_path}" \
        --original_config_file ./models/v1-inference.yaml \
        --dump_path ./converted/diffusers_model \
        --from_safetensors

  # Determine the model path to load for pipeline
  model_load_path = civitai_params.get("model_insert_path")
  if not model_load_path or not os.path.exists(model_load_path): # Add check for existence
      if civitai_params["status"] == "1" or civitai_params["status"] == "2":
          # If status is 1 or 2, and no insert path or path doesn't exist, use the downloaded/converted path
          model_load_path = "/content/converted/diffusers_model" if not civitai_params["sdxl"] else download_path
      elif civitai_params["status"] == "3":
          # If status is 3, use the default path for complete models
          model_load_path = "/content/converted/diffusers_model" if not civitai_params["sdxl"] else download_path # Assuming default path for complete models is the same

  #pipe = None
  if civitai_params["need_pipe"]:
    if civitai_params["sdxl"]:
      if os.path.isfile(model_load_path): # Check if it's a file
          from diffusers import StableDiffusionXLPipeline
          pipe_civitai = StableDiffusionXLPipeline.from_single_file(
              model_load_path, # Use model_load_path
              torch_dtype=torch.float16, # fp16モデルなので必須
              use_safetensors=True,      # safetensors形式であることを明示
              safety_checker=None,
              custom_pipeline="lpw_stable_diffusion"
          ).to("cuda")
      else: # Assume it's a directory
           from diffusers import StableDiffusionXLPipeline
           pipe_civitai = StableDiffusionXLPipeline.from_pretrained(
              model_load_path, # Use model_load_path
              torch_dtype=torch.float16, # fp16モデルなので必須
              safety_checker=None,
              custom_pipeline="lpw_stable_diffusion"
          ).to("cuda")
    else: # Not SDXL
      if os.path.isfile(model_load_path): # Check if it's a file
          from diffusers import StableDiffusionPipeline
          pipe_civitai = StableDiffusionPipeline.from_single_file(
              model_load_path, # Use model_load_path
              torch_dtype=torch.float16,
              use_safetensors=True, # Assuming single files are safetensors
              safety_checker=None,
              custom_pipeline="lpw_stable_diffusion"
          ).to("cuda")
      else: # Assume it's a directory
          from diffusers import StableDiffusionPipeline
          #pipe = StableDiffusionPipeline.from_pretrained(base_dir, torch_dtype=torch.float16).to("cuda")
          pipe_civitai = StableDiffusionPipeline.from_pretrained(
              model_load_path, # Use model_load_path
              torch_dtype=torch.float16,
              use_safetensors=False, # Assuming directories are not safetensors
              safety_checker=None,
              custom_pipeline="lpw_stable_diffusion"
            ).to("cuda")
    # パイプラインを作成した直後に以下を追加します
    if lora_path:
      pipe_civitai.load_lora_weights(lora_path, use_safetensors=True)

  modelName_civitai = model_load_path if model_load_path and os.path.exists(model_load_path) else (pipe_civitai.pretrained_model_name_or_path if hasattr(pipe_civitai, 'pretrained_model_name_or_path') else "unknown")

  print(mult)
  if civitai_params["submode"] == "1":
  #pipe.enable_xformers_memory_efficient_attention()
    for k in range(civitai_params["num"]):
      image = pipe_civitai(prompt=civitai_params["positive_prompt"],
          negative_prompt=civitai_params["negative_prompt"],
          num_inference_steps=civitai_params["steps"],
          width=Width,
          height=Height,
          guidance_scale=civitai_params["guidance"],
          max_embeddings_multiples=mult
      ).images[0]
      if drive_use:
        save_dir = os.path.join(
            drive_save,
            timeDir["y"]+"-"+timeDir["m"],
            timeDir["d"],
            timeDir["t"]
          )
        os.makedirs(save_dir, exist_ok=True)
        image.save(save_dir+"/"+str(k+1)+".png")

        # プロンプトとネガティブプロンプトを保存（1回目のみ）
        if k == 0:
          with open(os.path.join(save_dir, "info.txt"), "w", encoding="utf-8") as f:
            f.write("t2i\n\n")
            f.write("Width,Height:\n"+str(Width)+","+str(Height)+"\n\n")
            if civitai_params["sdxl"]:
              f.write("Model:\nCivitAI(SDXL) / "+modelName_civitai+"\n\n")
            else:
              f.write("Model:\nCivitAI / "+modelName_civitai+"\n\n")
            f.write("Num_Inference_Steps:\n"+str(civitai_params["steps"])+"\n\n")
            f.write("Positive Prompt:"+str(pos_num)+"\n" + civitai_params["positive_prompt"] + "\n\n")
            f.write("Negative Prompt:"+str(pos_num)+"\n" + civitai_params["negative_prompt"])

      else:
        os.makedirs("output/t2i/"+timestamp+"/"+civitai_params["positive_prompt"][0:10], exist_ok=True)
        image.save("output/t2i/"+timestamp+"/"+civitai_params["positive_prompt"][0:10]+"/"+str(k+1)+".png")
    print("画像を保存しました: output/t2i/"+timestamp+"/"+civitai_params["positive_prompt"][0:10])
  else:
    init_image = Image.open(civitai_params["image_path"]).convert("RGB")
    img_width, img_height = init_image.size
    if not OriginSize:
      init_image = ImageOps.fit(init_image, (Width, Height), method=Image.Resampling.LANCZOS)
      img_width = Width
      img_height = Height
    #display(init_image)
    for k in range(civitai_params["num"]):
      image = pipe_civitai(prompt=civitai_params["positive_prompt"],
          negative_prompt=civitai_params["negative_prompt"],
          image=init_image,
          num_inference_steps=civitai_params["steps"],
          strength=civitai_params["strength"],
          width=img_width,
          height=img_height,
          guidance_scale=civitai_params["guidance"],
          max_embeddings_multiples=mult
      ).images[0]
      if drive_use:
        save_dir = os.path.join(
            drive_save,
            timeDir["y"]+"-"+timeDir["m"],
            timeDir["d"],
            timeDir["t"]
          )
        os.makedirs(save_dir, exist_ok=True)
        image.save(save_dir+"/"+str(k+1)+".png")

        # プロンプトとネガティブプロンプトを保存（1回目のみ）
        if k == 0:
          init_image.save(save_dir+"/origin.png")
          with open(os.path.join(save_dir, "info.txt"), "w", encoding="utf-8") as f:
            f.write("i2i\n\n")
            if civitai_params["sdxl"]:
              f.write("Model:\nCivitAI(SDXL) / "+modelName_civitai+"\n\n")
            else:
              f.write("Model:\nCivitAI / "+modelName_civitai+"\n\n")
            f.write("Num_Inference_Steps:\n"+str(civitai_params["steps"])+"\n\n")
            f.write("Strength:\n"+str(civitai_params["strength"])+"\n\n")
            f.write("Size:\nw..."+str(img_width)+",h..."+str(img_height)+"\n\n")
            f.write("Positive Prompt:"+str(pos_num)+"\n" + civitai_params["positive_prompt"] + "\n\n")
            f.write("Negative Prompt:"+str(neg_num)+"\n" + civitai_params["negative_prompt"])
      else:
        os.makedirs("output/i2i/"+timestamp+"/"+civitai_params["positive_prompt"][0:10], exist_ok=True)
        image.save("output/i2i/"+timestamp+"/"+civitai_params["positive_prompt"][0:10]+"/"+str(k+1)+".png")
    print("画像を保存しました: output/i2i/"+timestamp+"/"+civitai_params["positive_prompt"][0:10])

