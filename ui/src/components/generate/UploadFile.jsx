import dynamic from "next/dynamic";
import { useEffect, useRef, useState } from "react";
import { toast } from "react-toastify";
import Dropzone from "@components/components/Dropzone";
import { IconLink } from "@tabler/icons-react";
import Dropdown from "@components/components/Dropdown";
import { AVAILABLE_MODELS, LANGUAGES_PER_MODEL } from "@components/constants";
import { getRequestParamsForModel } from "@components/utils";
import { formatLanguagesForDropdownOptions } from "@components/dropdownUtils";
import useLocalStorage from "@components/hooks/useLocalStorage";

const UploadFile = ({
  uploadedFile,
  setUploadedFile,
  setYoutubeLink,
  youtubeLink,
  setIsBeingGenerated,
  setTranscribed,
  setrequestSentToAPI,
  isLocalFile,
  setYoutubeTitle,
  selectedModel,
  setSelectedModel,
  setIsSubtitleGenerated,
}) => {
  const [turnOnAdvanceOptions, setTurnOnAdvanceOptions] = useState(false);
  const [targetLanguage, setTargetLanguage] = useLocalStorage(
    "target-language",
    ""
  );

  const [disabled, setDisabled] = useState(true);
  const [submitCounter, setSubmitCounter] = useState(0);

  function storeFileToLocalStorage(file) {
    const items = JSON.parse(localStorage.getItem("file"));
    if (!items) {
      localStorage.setItem("file", JSON.stringify([file]));
    } else {
      items.push(file);
      localStorage.setItem("file", JSON.stringify(items));
    }
  }

  useEffect(() => {
    if ((uploadedFile || youtubeLink) && targetLanguage && !isLocalFile) {
      setDisabled(false);
    } else setDisabled(true);
  }, [uploadedFile, targetLanguage, youtubeLink]);

  useEffect(() => {
    uploadButtonRef?.current?.scrollIntoView({
      behavior: "smooth",
      block: "end",
      inline: "nearest",
    });
  }, [turnOnAdvanceOptions]);

  function reset(state) {
    setDisabled(state);
    setrequestSentToAPI(state);
  }

  const handleInputChange = (event) => {
    const value = event.target.value;
    setYoutubeLink(value);
  };

  async function youtubeVideoTitle() {
    const data = await fetch(
      `https://noembed.com/embed?dataType=json&url=${youtubeLink}`
    );
    const jsonData = await data.json();
    const title = jsonData.title;
    setYoutubeTitle(title);
    return title;
  }

  async function handleSubmit() {
    if (uploadedFile && youtubeLink) {
      return toast.error("Cannot upload both file and YouTube link");
    }
    reset(true);
    setSubmitCounter(submitCounter + 1);
    if (submitCounter > 0) setTranscribed([]);

    const handleServerResponse = (jsonData = {}) => {
      switch (jsonData.type) {
        case "language_detection":
          const language_identified = jsonData["data"];
          toast.info("Language identified as " + language_identified, {
            autoClose: 8000,
          });
          return true;
        case "info":
          toast.info(jsonData.data, { autoClose: 8000 });
          return true;
        case "error":
          toast.error(jsonData.data);
          return true;
        default:
          // Handle flat transcription segments
          if (jsonData.start !== undefined && jsonData.end !== undefined && jsonData.text !== undefined) {
            setTranscribed((transcribed) => [...transcribed, jsonData]);
            return true;
          }
          return false;
      }
    };

    const toastId = toast.info("Uploading..");
    try {
      const { url, requestData } = await getRequestParamsForModel(
        uploadedFile,
        youtubeLink,
        targetLanguage,
        selectedModel
      );

      const res = await fetch(url, {
        method: "POST",
        body: requestData,
      });

      if (!res.ok) throw new Error("Server error during transcription");

      toast.update(toastId, { render: "Transcribing..", type: "info" });
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let transcription = [];

      while (true) {
        const { done, value } = await reader.read();
        setIsBeingGenerated(!done);
        if (done) {
          setIsSubtitleGenerated(true);
          break;
        }
        try {
          const decodedValue = decoder.decode(value);
          // Split the response into individual JSON objects
          const jsonObjects = decodedValue.match(/{[^}]+}/g) || [];
          for (const objString of jsonObjects) {
            try {
              const jsonData = JSON.parse(objString);
              const isServerMsg = handleServerResponse(jsonData);
              if (!isServerMsg) {
                // Handle flat transcription segments
                transcription.push(jsonData);
              }
            } catch (parseError) {
              console.error("Error parsing JSON:", parseError);
            }
          }
        } catch (error) {
          console.error("Error processing stream:", error);
        }
      }

      toast.update(toastId, {
        render: "Successfully transcribed",
        type: "success",
      });

      const file = {
        filename: uploadedFile?.path ?? "Local Audio",
        link: youtubeLink,
        size: uploadedFile?.size,
        transcribedData: transcription,
        uploadDate: new Date(),
        model: selectedModel,
        targetLanguage: targetLanguage,
      };
      storeFileToLocalStorage(file);
    } catch (err) {
      toast.update(toastId, {
        type: "error",
        render:
          "There seems to be some error in transcribing. Please try again later with a different file.",
      });
    } finally {
      reset(false);
    }
  }

  function toggleAdvanceMode() {
    setTurnOnAdvanceOptions((prevState) => {
      if (prevState) {
        setSelectedModel("fasterWhisper");
      }
      return !prevState;
    });
  }

  const handleModelChange = (item) => {
    setSelectedModel(item);
  };

  const uploadButtonRef = useRef();
  return (
    <div className="overflow-hidden overflow-y-auto  lg:px-4">
      <div>
        <h2 className="text-3xl font-medium">Upload a File</h2>
        <p className="font-xl text-gray-500 font-medium mt-2">
          Upload an audio file to generate subtitles
        </p>
      </div>
      <div className="h-64">
        <Dropzone
          setUploadedFile={setUploadedFile}
          uploadedFile={uploadedFile}
        />
      </div>
      <div className="mt-2">
        <a
          href="/blog/our-recommendations"
          target="_blank"
          className="link link-primary"
        >
          Learn about our model recommendation
        </a>
      </div>
      <div className="flex items-center justify-end ">
        <div className="form-control">
          <label className="label cursor-pointer">
            <span className="label-text mx-2 font-medium">Advance Options</span>
            <input
              type="checkbox"
              className="toggle toggle-primary"
              defaultValue={turnOnAdvanceOptions}
              onClick={toggleAdvanceMode}
            />
          </label>
        </div>
      </div>
      {turnOnAdvanceOptions ? (
        <div className="space-y-5">
          <Dropdown
            onChange={handleModelChange}
            label="Generation Model"
            options={AVAILABLE_MODELS}
            defaultOption="Select Model"
            selectedVal={selectedModel}
          />
          <Dropdown
            onChange={(item) => setTargetLanguage(item)}
            label="Subtitle Language"
            options={formatLanguagesForDropdownOptions(selectedModel)}
            keyName="target-language"
            defaultOption="Select Language"
            selectedVal={targetLanguage}
          />
          <div className="hidden">
            <p className="font-medium text-wrap">Prompt:</p>
            <p className="font-medium text-xs text-gray-500 mt-[-5px]">
              Optional
            </p>
            <textarea
              name=""
              id=""
              className="resize-none p-2 w-full border-2 rounded-md outline-none"
              rows="10"
              placeholder="enter your prompt here......"
            ></textarea>
          </div>
        </div>
      ) : (
        <>
          <Dropdown
            onChange={(item) => setTargetLanguage(item)}
            label="Subtitle Language"
            options={formatLanguagesForDropdownOptions(selectedModel)}
            keyName="target-language"
            defaultOption="Select Language"
            selectedVal={targetLanguage}
          />
        </>
      )}
      <button
        ref={uploadButtonRef}
        disabled={disabled}
        onClick={handleSubmit}
        className={` ${
          disabled
            ? "bg-gray-400 hover:cursor-not-allowed"
            : "bg-primary-900 hover:cursor-pointer"
        } w-full mt-5 text-white py-2 rounded-md text-lg font-medium transition-all duration-300 flex items-center justify-center`}
      >
        Generate
      </button>
    </div>
  );
};

export default dynamic(() => Promise.resolve(UploadFile), { ssr: false });