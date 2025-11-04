// Copyright (c) 2022 National Instruments
// SPDX-License-Identifier: MIT

#include "RdmaCommon.h"
#include <boost/thread.hpp>
#include <mutex>
#include <assert.h>
#include <valgrind.h>
#include <atomic>
#include "EventManager.h"
#include "ThreadUtility.h"

// File local variables
static std::mutex eventChannelMutex;
static rdma_event_channel* eventChannel = nullptr;
static boost::thread eventThread;
static EventManager eventManager;
static std::atomic<bool> shutdownInProgress{false};

void EventChannelThread(rdma_event_channel* eventChannel)
{
    try {
        while (true) {
            rdma_cm_event* event = nullptr;
            HandleError(rdma_get_cm_event(eventChannel, &event));
            try {
                reinterpret_cast<iEventHandler*>(event->id->context)->SignalEvent(event);
            } catch (std::exception& e) {
                rdma_ack_cm_event(event);
                throw;
            }
            rdma_ack_cm_event(event);
        }
    } catch (const RdmaException& e) {
        throw;
    } catch (std::exception& e) {
        RDMA_THROW(easyrdma_Error_InternalError);
    }
}

rdma_event_channel* GetEventChannel()
{
    std::unique_lock<std::mutex> guard(eventChannelMutex);
    if (!eventChannel) {
        eventChannel = rdma_create_event_channel();
        HandleErrorFromPointer(eventChannel);
        eventThread = CreatePriorityThread(boost::bind(EventChannelThread, eventChannel), kThreadPriority::Normal, "EventHandler");

        // The way the event mechanism in rdma_cm works, there is not a good way to abort the blocking wait on the event channel.
        // Since we share a singleton event channel among all sessions, we will simply detach this thread and let process destruction
        // deal with it. Possible options for improving it would be to access the internals of the event structure and find the fd it is
        // reading from and use poll() combined with a separate signal fd to abort it
        if (eventThread.joinable()) {
            eventThread.detach();
        }
    }
    return eventChannel;
}

EventManager& GetEventManager()
{
    return eventManager;
}

bool IsShuttingDown()
{
    return shutdownInProgress.load();
}

// Helper class to detect global shutdown
struct ShutdownDetector
{
    ~ShutdownDetector()
    {
        shutdownInProgress.store(true);
    }
};
static ShutdownDetector shutdownDetector;

bool IsValgrindRunning()
{
    static bool running = RUNNING_ON_VALGRIND;
    return running;
}