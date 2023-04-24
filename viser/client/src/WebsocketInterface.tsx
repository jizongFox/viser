import { pack, unpack } from "msgpackr";
import React, { MutableRefObject, useContext } from "react";
import * as THREE from "three";
import AwaitLock from "await-lock";
import { TextureLoader } from "three";

import { SceneNode } from "./SceneTree";
import { CoordinateFrame, CameraFrustum } from "./ThreeAssets";
import { Message } from "./WebsocketMessages";
import { syncSearchParamServer } from "./SearchParamsUtils";
import { PivotControls } from "@react-three/drei";
import { ViewerContext } from ".";
import { useThree } from "@react-three/fiber";

/** Send message over websocket. */
export function sendWebsocketMessage(
  websocketRef: MutableRefObject<WebSocket | null>,
  message: Message
) {
  if (websocketRef.current === null) return;
  websocketRef.current.send(pack(message));
}

/** Returns a function for sending messages, with automatic throttling. */
export function makeThrottledMessageSender(
  websocketRef: MutableRefObject<WebSocket | null>,
  throttleMilliseconds: number
) {
  let readyToSend = true;
  let stale = false;
  let latestMessage: Message | null = null;

  function send(message: Message) {
    if (websocketRef.current === null) return;
    latestMessage = message;
    if (readyToSend) {
      websocketRef.current.send(pack(message));
      stale = false;
      readyToSend = false;

      setTimeout(() => {
        readyToSend = true;
        if (!stale) return;
        latestMessage && send(latestMessage);
      }, throttleMilliseconds);
    } else {
      stale = true;
    }
  }
  return send;
}

/** Type guard for threejs textures. Meant to be used with `scene.background`. */
function isTexture(
  background: THREE.Color | THREE.Texture | THREE.CubeTexture | null
): background is THREE.Texture {
  return (
    background !== null && (background as THREE.Texture).isTexture !== undefined
  );
}

/** Returns a handler for all incoming messages. */
function useMessageHandler() {
  const viewer = useContext(ViewerContext)!;
  const scene = useThree((state) => state.scene);

  const removeSceneNode = viewer.useSceneTree((state) => state.removeSceneNode);
  const resetScene = viewer.useSceneTree((state) => state.resetScene);
  const addSceneNode = viewer.useSceneTree((state) => state.addSceneNode);
  const addGui = viewer.useGui((state) => state.addGui);
  const removeGui = viewer.useGui((state) => state.removeGui);
  const guiSet = viewer.useGui((state) => state.guiSet);
  const setOrientation = viewer.useSceneTree((state) => state.setOrientation);
  const setPosition = viewer.useSceneTree((state) => state.setPosition);
  const setVisibility = viewer.useSceneTree((state) => state.setVisibility);

  // Same as addSceneNode, but make a parent in the form of a dummy coordinate
  // frame if it doesn't exist yet.
  function addSceneNodeMakeParents(node: SceneNode) {
    const nodeFromName = viewer.useSceneTree.getState().nodeFromName;
    const parent_name = node.name.split("/").slice(0, -1).join("/");
    if (!(parent_name in nodeFromName)) {
      addSceneNodeMakeParents(
        new SceneNode(parent_name, (ref) => (
          <CoordinateFrame ref={ref} show_axes={false} />
        ))
      );
    }
    addSceneNode(node);
  }

  // Return message handler.
  return (message: Message) => {
    switch (message.type) {
      // Add a coordinate frame.
      case "FrameMessage": {
        addSceneNodeMakeParents(
          new SceneNode(message.name, (ref) => (
            <CoordinateFrame
              ref={ref}
              position={new THREE.Vector3().fromArray(message.position)}
              quaternion={
                new THREE.Quaternion(
                  message.wxyz[1],
                  message.wxyz[2],
                  message.wxyz[3],
                  message.wxyz[0]
                )
              }
              show_axes={message.show_axes}
              axes_length={message.axes_length}
              axes_radius={message.axes_radius}
            />
          ))
        );
        break;
      }
      // Add a point cloud.
      case "PointCloudMessage": {
        const geometry = new THREE.BufferGeometry();
        const pointCloudMaterial = new THREE.PointsMaterial({
          size: message.point_size,
          vertexColors: true,
        });

        // Reinterpret cast: uint8 buffer => float32 for positions.
        geometry.setAttribute(
          "position",
          new THREE.Float32BufferAttribute(
            new Float32Array(
              message.position.buffer.slice(
                message.position.byteOffset,
                message.position.byteOffset + message.position.byteLength
              )
            ),
            3
          )
        );
        geometry.computeBoundingSphere();

        // Wrap uint8 buffer for colors. Note that we need to set normalized=true.
        geometry.setAttribute(
          "color",
          new THREE.Uint8BufferAttribute(message.color, 3, true)
        );

        addSceneNodeMakeParents(
          new SceneNode(message.name, (ref) => (
            <points
              ref={ref}
              geometry={geometry}
              material={pointCloudMaterial}
            />
          ))
        );
        break;
      }
      // Add mesh
      case "MeshMessage": {
        const geometry = new THREE.BufferGeometry();
        // TODO(hangg): Should expose color as well.
        const material = new THREE.MeshStandardMaterial({
          color: message.color,
          wireframe: message.wireframe,
        });
        geometry.setAttribute(
          "position",
          new THREE.Float32BufferAttribute(
            new Float32Array(
              message.vertices.buffer.slice(
                message.vertices.byteOffset,
                message.vertices.byteOffset + message.vertices.byteLength
              )
            ),
            3
          )
        );
        geometry.setIndex(
          new THREE.Uint32BufferAttribute(
            new Uint32Array(
              message.faces.buffer.slice(
                message.faces.byteOffset,
                message.faces.byteOffset + message.faces.byteLength
              )
            ),
            1
          )
        );
        geometry.computeVertexNormals();
        geometry.computeBoundingSphere();
        addSceneNodeMakeParents(
          new SceneNode(message.name, (ref) => (
            <mesh ref={ref} geometry={geometry} material={material} />
          ))
        );
        break;
      }
      // Add a camera frustum.
      case "CameraFrustumMessage": {
        addSceneNodeMakeParents(
          new SceneNode(message.name, (ref) => (
            <CameraFrustum
              ref={ref}
              fov={message.fov}
              aspect={message.aspect}
              scale={message.scale}
              color={message.color}
            ></CameraFrustum>
          ))
        );
        break;
      }
      case "TransformControlsMessage": {
        const name = message.name;
        const sendDragMessage = makeThrottledMessageSender(
          viewer.websocketRef,
          25
        );
        addSceneNodeMakeParents(
          new SceneNode(message.name, (ref) => (
            <PivotControls
              ref={ref}
              scale={message.scale}
              lineWidth={message.line_width}
              fixed={message.fixed}
              autoTransform={message.auto_transform}
              activeAxes={message.active_axes}
              disableAxes={message.disable_axes}
              disableSliders={message.disable_sliders}
              disableRotations={message.disable_rotations}
              translationLimits={message.translation_limits}
              rotationLimits={message.rotation_limits}
              depthTest={message.depth_test}
              opacity={message.opacity}
              onDrag={(l) => {
                const wxyz = new THREE.Quaternion();
                wxyz.setFromRotationMatrix(l);
                const position = new THREE.Vector3().setFromMatrixPosition(l);
                sendDragMessage({
                  type: "TransformControlsUpdateMessage",
                  name: name,
                  wxyz: [wxyz.w, wxyz.x, wxyz.y, wxyz.z],
                  position: position.toArray(),
                });
              }}
            />
          ))
        );
        break;
      }
      case "SetCameraOrientationMessage": {
        const camera = viewer.globalCameras.current!.cameras[viewer.panelKey];
        const cameraControls =
          viewer.globalCameras.current!.cameraControlRefs[viewer.panelKey]
            .current!;

        const R_world_camera = new THREE.Quaternion(
          message.wxyz[1],
          message.wxyz[2],
          message.wxyz[3],
          message.wxyz[0]
        );

        const R_cam_threecam = new THREE.Quaternion();
        const R_threeworld_world = new THREE.Quaternion();
        R_cam_threecam.setFromEuler(new THREE.Euler(-Math.PI, 0.0, 0.0));
        R_threeworld_world.setFromEuler(
          new THREE.Euler(-Math.PI / 2.0, 0.0, 0.0)
        );
        const R_threeworld_threecamera = R_threeworld_world.clone()
          .multiply(R_world_camera)
          .multiply(R_cam_threecam);

        // Compute a look-at point, while holding the orbit distance constant.
        const position = cameraControls.getPosition(new THREE.Vector3());
        const currentTargetDist = cameraControls
          .getTarget(new THREE.Vector3())
          .sub(position)
          .length();
        const target = new THREE.Vector3(0.0, 0.0, -currentTargetDist)
          .applyQuaternion(R_threeworld_threecamera)
          .add(position);

        // Compute and apply a new up direction.
        // The goal here is simple: set the up direction s.t. the camera roll is correct,
        // but keep the up direction as close to a world-frame vertical as possible.
        const eps = 1e-7;
        const facePlane = target.clone().sub(position).setY(eps).normalize();
        const up = new THREE.Vector3(eps, 1.0, eps)
          .projectOnPlane(facePlane)
          .normalize();
        camera.up.set(up.x, up.y, up.z);

        cameraControls.setLookAt(
          position.x,
          position.y,
          position.z,
          target.x,
          target.y,
          target.z
        );
        break;
      }
      case "SetCameraPositionMessage": {
        const cameraControls =
          viewer.globalCameras.current!.cameraControlRefs[viewer.panelKey]
            .current!;

        // When setting camera position, we're going to shift the look-at point by the same amount as the position.
        // This will hold the camera orientation constant.
        const position_cur = new THREE.Vector3();
        cameraControls.getPosition(position_cur);

        const position_cmd = new THREE.Vector3(
          message.position[0],
          message.position[1],
          message.position[2]
        );
        const R_worldthree_world = new THREE.Quaternion();
        R_worldthree_world.setFromEuler(
          new THREE.Euler(-Math.PI / 2.0, 0.0, 0.0)
        );
        position_cmd.applyQuaternion(R_worldthree_world);

        const target = cameraControls
          .getTarget(new THREE.Vector3())
          .add(position_cmd)
          .sub(position_cur);

        cameraControls.setLookAt(
          position_cmd.x,
          position_cmd.y,
          position_cmd.z,
          target.x,
          target.y,
          target.z
        );
        break;
      }
      case "SetCameraFovMessage": {
        const camera = viewer.globalCameras.current!.cameras[viewer.panelKey];
        // tan(fov / 2.0) = 0.5 * film height / focal length
        // focal length = 0.5 * film height / tan(fov / 2.0)
        camera.setFocalLength(
          (0.5 * camera.getFilmHeight()) / Math.tan(message.fov / 2.0)
        );
        break;
      }
      case "SetOrientationMessage": {
        setOrientation(
          message.name,
          new THREE.Quaternion(
            message.wxyz[1],
            message.wxyz[2],
            message.wxyz[3],
            message.wxyz[0]
          )
        );
        break;
      }
      case "SetPositionMessage": {
        setPosition(
          message.name,
          new THREE.Vector3(
            message.position[0],
            message.position[1],
            message.position[2]
          )
        );
        break;
      }
      // Add a background image.
      case "BackgroundImageMessage": {
        const oldBackground = scene.background;
        const texture = new TextureLoader().load(
          `data:${message.media_type};base64,${message.base64_data}`
        );
        scene.background = texture;
        scene.background.encoding = THREE.sRGBEncoding;

        if (isTexture(oldBackground)) oldBackground.dispose();
        break;
      }
      // Add an image.
      case "ImageMessage": {
        // It's important that we load the texture outside of the node
        // construction callback; this prevents flickering by ensuring that the
        // texture is ready before the scene tree updates.
        const colorMap = new TextureLoader().load(
          `data:${message.media_type};base64,${message.base64_data}`
        );
        addSceneNodeMakeParents(
          new SceneNode(message.name, (ref) => {
            return (
              <mesh ref={ref}>
                <planeGeometry
                  attach="geometry"
                  args={[message.render_width, message.render_height]}
                />
                <meshBasicMaterial
                  attach="material"
                  transparent={true}
                  side={THREE.DoubleSide}
                  map={colorMap}
                />
              </mesh>
            );
          })
        );
        break;
      }
      // Remove a scene node by name.
      case "RemoveSceneNodeMessage": {
        console.log("Removing scene node:", message.name);
        removeSceneNode(message.name);
        break;
      }
      // Set the visibility of a particular scene node.
      case "SetSceneNodeVisibilityMessage": {
        setVisibility(message.name, message.visible);
        break;
      }
      // Reset the entire scene, removing all scene nodes.
      case "ResetSceneMessage": {
        resetScene();

        const oldBackground = scene.background;
        scene.background = null;
        if (isTexture(oldBackground)) oldBackground.dispose();

        viewer.useGui.setState({ backgroundAvailable: false });
        break;
      }
      // Add a GUI input.
      case "GuiAddMessage": {
        addGui(message.name, {
          levaConf: message.leva_conf,
          folderLabels: message.folder_labels,
          visible: true,
        });
        break;
      }
      // Set the value of a GUI input.
      case "GuiSetValueMessage": {
        guiSet(message.name, message.value);
        break;
      }
      // Set the hidden state of a GUI input.
      case "GuiSetVisibleMessage": {
        const currentConf =
          viewer.useGui.getState().guiConfigFromName[message.name];
        if (currentConf !== undefined) {
          addGui(message.name, {
            ...currentConf,
            visible: message.visible,
          });
        }
        break;
      }
      // Add a GUI input.
      case "GuiSetLevaConfMessage": {
        const currentConf =
          viewer.useGui.getState().guiConfigFromName[message.name];
        if (currentConf !== undefined) {
          addGui(message.name, {
            ...currentConf,
            levaConf: message.leva_conf,
          });
        }
        break;
      }
      // Remove a GUI input.
      case "GuiRemoveMessage": {
        removeGui(message.name);
        break;
      }
      default: {
        console.log("Received message did not match any known types:", message);
        break;
      }
    }
  };
}

/** Component for handling websocket connections. */
export default function WebsocketInterface() {
  const viewer = useContext(ViewerContext)!;
  const handleMessage = useMessageHandler();

  const server = viewer.useGui((state) => state.server);
  const resetGui = viewer.useGui((state) => state.resetGui);

  syncSearchParamServer(viewer.panelKey, server);

  React.useEffect(() => {
    // Lock for making sure messages are handled in order.
    const orderLock = new AwaitLock();

    let ws: null | WebSocket = null;
    let done = false;

    function tryConnect(): void {
      if (done) return;

      ws = new WebSocket(server);

      ws.onopen = () => {
        console.log("Connected!" + server);
        viewer.websocketRef.current = ws;
        viewer.useGui.setState({ websocketConnected: true });
      };

      ws.onclose = () => {
        console.log("Disconnected! " + server);
        viewer.websocketRef.current = null;
        viewer.useGui.setState({ websocketConnected: false });
        if (viewer.useGui.getState().guiNames.length > 0) resetGui();

        // Try to reconnect.
        timeout = setTimeout(tryConnect, 1000);
      };

      const messageQueue: Message[] = [];
      const messageGroup: Message[] = [];
      let grouping = false;

      // Handle batches of messages at 200Hz. This helps prevent some React errors from interrupting renders.
      setInterval(() => {
        const numMessages = messageQueue.length;
        const processBatch = messageQueue.slice(0, numMessages);
        messageQueue.splice(0, numMessages);
        processBatch.forEach(handleMessage);
      }, 5);

      ws.onmessage = async (event) => {
        // Reduce websocket backpressure.
        const messagePromise = new Promise<Message>((resolve) => {
          (event.data.arrayBuffer() as Promise<ArrayBuffer>).then((buffer) => {
            resolve(unpack(new Uint8Array(buffer)) as Message);
          });
        });

        // Try our best to handle messages in order. If this takes more than 1 second, we give up. :)
        await orderLock.acquireAsync({ timeout: 1000 }).catch(() => {
          console.log("Order lock timed out.");
          orderLock.release();
        });
        try {
          const message = await messagePromise;
          if (message.type === "MessageGroupStart") grouping = true;
          else if (message.type === "MessageGroupEnd") {
            messageQueue.push(...messageGroup);
            messageGroup.length = 0;
            grouping = false;
          } else if (grouping) {
            messageGroup.push(message);
          } else {
            messageQueue.push(message);
          }
        } finally {
          orderLock.acquired && orderLock.release();
        }
      };
    }

    let timeout = setTimeout(tryConnect, 500);
    return () => {
      done = true;
      clearTimeout(timeout);
      viewer.useGui.setState({ websocketConnected: false });
      ws && ws.close();
      clearTimeout(timeout);
    };
  }, [server, handleMessage, resetGui]);

  return <></>;
}